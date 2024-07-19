def _train_update(device, step, loss, tracker, epoch, writer):
  test_utils.print_training_update(
      device,
      step,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      epoch,
      summary_writer=writer)


def train_imagenet():
    if FLAGS.pjrt_distributed:
        dist.init_process_group('xla', init_method='xla://')

    train_sampler, test_sampler = None, None
    if xm.xrt_world_size() > 1:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)
      test_sampler = torch.utils.data.distributed.DistributedSampler(
          test_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        sampler=train_sampler,
        drop_last=FLAGS.drop_last,
        shuffle=False if train_sampler else True,
        persistent_workers=True,
        num_workers=FLAGS.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=FLAGS.test_set_batch_size,
        sampler=test_sampler,
        drop_last=FLAGS.drop_last,
        shuffle=False,
        persistent_workers=True,
        num_workers=FLAGS.num_workers)

  torch.manual_seed(42)

  device = xm.xla_device()
  
  
 ###############################################################################################################################################################################################FSDP wrap#################################################################################################################################################################################### 
  
  # Automatic wrapping sub-modules with inner FSDP
  auto_wrap_policy = None
  auto_wrapper_callable = None
  if FLAGS.auto_wrap_policy == "size_based":
    # auto-wrap all sub-modules with a certain number of parameters (default 1e6)
    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=FLAGS.auto_wrap_min_num_params)

   if FLAGS.use_gradient_checkpointing:
    # Apply gradient checkpointing to auto-wrapped sub-modules if specified
    auto_wrapper_callable = lambda m, *args, **kwargs: FSDP(
        checkpoint_module(m), *args, **kwargs)

  fsdp_wrap = lambda m: FSDP(
      m,
      compute_dtype=getattr(torch, FLAGS.compute_dtype),
      fp32_reduce_scatter=FLAGS.fp32_reduce_scatter,
      flatten_parameters=FLAGS.flatten_parameters,
      shard_param_on_dim_0=FLAGS.shard_param_on_dim_0,
      pin_layout_in_collective_ops=FLAGS.pin_layout_in_collective_ops,
      auto_wrap_policy=auto_wrap_policy,
      auto_wrapper_callable=auto_wrapper_callable)
  
  # Manually wrapping sub-modules with inner FSDP (if not using auto-wrap)
  # (in this case, the sub-modules should be wrapped before the base model)
  if FLAGS.use_nested_fsdp:
    assert FLAGS.auto_wrap_policy == "none", \
        "--use_nested_fsdp is for manual nested wrapping should only be used" \
        " without auto-wrapping"
    # You may wrap all, a subset, or none of the sub-modules with inner FSDPs
    # - to implement ZeRO-2, wrap none of the sub-modules
    # - to implement ZeRO-3, wrap all of the sub-modules (nested FSDP)
    # - you may wrap sub-modules at different granularity (e.g. at each resnet
    #   stage or each residual block or each conv layer).
    # Here we apply inner FSDP at the level of child modules for ZeRO-3, which
    # corresponds to different stages in resnet (i.e. Stage 1 to 5).
    # Apply gradient checkpointing to nested-wrapped sub-modules if specified
    grad_ckpt_wrap = checkpoint_module if FLAGS.use_gradient_checkpointing else (
        lambda x: x)
    for submodule_name, submodule in model.named_children():
      if sum(p.numel() for p in submodule.parameters()) == 0:
        # Skip those submodules without parameters (i.e. no need to shard them)
        continue
      # Note: wrap with `checkpoint_module` first BEFORE wrapping with FSDP
      m_fsdp = fsdp_wrap(grad_ckpt_wrap(getattr(model, submodule_name)))
      setattr(model, submodule_name, m_fsdp)

###########################################################################################################################################################################################################################################################################################################################################################################################################################################

  model = fsdp_wrap(model)
    

  optimizer = optim.SGD(
      model.parameters(),
      lr=FLAGS.lr,
      momentum=FLAGS.momentum,
      weight_decay=1e-4)
  
  
  num_training_steps_per_epoch = train_dataset_len // (
      FLAGS.batch_size * xm.xrt_world_size())
  lr_scheduler = None
  loss_fn = nn.CrossEntropyLoss()
  

  def train_loop_fn(loader, epoch):
    tracker = xm.RateTracker()
    model.train()
    for step, (data, target) in enumerate(loader):
      with xp.StepTrace('train_imagenet', step_num=step):
        with xp.Trace('build_graph'):
          optimizer.zero_grad()
          output = model(data)
          loss = loss_fn(output, target)
          loss.backward()
          optimizer.step()  # do not reduce gradients on sharded params
          tracker.add(FLAGS.batch_size)
          if lr_scheduler:
            lr_scheduler.step()
        if step % FLAGS.log_steps == 0:
          xm.add_step_closure(
              _train_update, args=(device, step, loss, tracker, epoch, writer))

  def test_loop_fn(loader, epoch):
    total_samples, correct = 0, 0
    model.eval()
    for step, (data, target) in enumerate(loader):
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum()
      total_samples += data.size()[0]
      if step % FLAGS.log_steps == 0:
        xm.add_step_closure(
            test_utils.print_test_update, args=(device, None, epoch, step))
    accuracy = 100.0 * correct.item() / total_samples
    accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
    return accuracy

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  accuracy, max_accuracy = 0.0, 0.0
  
  for epoch in range(1):
    xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    
    train_loop_fn(train_device_loader, epoch)
    
    xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
    
    run_eval = ((not FLAGS.test_only_at_end and
                 epoch % FLAGS.eval_interval == 0) or epoch == FLAGS.num_epochs)
    
    
    if run_eval:
      with torch.no_grad():
        accuracy = test_loop_fn(test_device_loader, epoch)

      xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
          epoch, test_utils.now(), accuracy))
      
      max_accuracy = max(accuracy, max_accuracy)

  xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))


def _mp_fn(index, flags):
  global FLAGS
  FLAGS = flags
  torch.set_default_dtype(torch.float32)
  accuracy = train_imagenet()


if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)