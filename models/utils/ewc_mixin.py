def compute_fisher(self, dataset, num_samples=200):
    """
    Compute Fisher Information Matrix.
    FIXED: Handles Mammoth's dataset indexing correctly.
    """
    print(f"[EWC] Computing Fisher for Task {self.task_count + 1}...")
    
    # Initialize new Fisher
    new_fisher = {}
    for name, param in self.net.named_parameters():
        if param.requires_grad:
            new_fisher[name] = torch.zeros_like(param)
    
    self.net.eval()
    
    # ========== FIX: Get CURRENT task data (not next task!) ==========
    try:
        # Mammoth datasets have current task at index i-1 after training
        # Because i gets incremented BEFORE end_task() is called
        current_task_id = dataset.i - 1  # Go back to task we just trained
        
        if current_task_id < 0:
            current_task_id = 0
        
        print(f"[EWC] Accessing task {current_task_id} for Fisher computation")
        
        # Temporarily set dataset to current task
        original_i = dataset.i
        dataset.i = current_task_id
        
        # Get train loader
        train_loader, _ = dataset.get_data_loaders()
        
        # Restore dataset index
        dataset.i = original_i
        
    except Exception as e:
        print(f"[EWC] Error getting data loader: {e}")
        print(f"[EWC] Skipping Fisher computation for this task")
        self.task_count += 1
        self.net.train()
        return
    
    # ========== Sample and compute Fisher ==========
    samples_seen = 0
    batch_count = 0
    max_batches = max(1, num_samples // 32)
    
    try:
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            
            # Unpack batch (Mammoth format)
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) >= 3:
                    inputs, labels, _ = batch_data[:3]
                else:
                    inputs, labels = batch_data[:2]
            else:
                print(f"[EWC] Unexpected batch format, skipping")
                continue
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward + backward
            outputs = self.net(inputs)
            loss = nn.functional.cross_entropy(outputs, labels)
            
            self.opt.zero_grad()
            loss.backward()
            
            # Accumulate gradients
            for name, param in self.net.named_parameters():
                if param.requires_grad and param.grad is not None:
                    new_fisher[name] += param.grad.pow(2) * inputs.size(0)
            
            samples_seen += inputs.size(0)
            batch_count += 1
            
    except Exception as e:
        print(f"[EWC] Error during Fisher computation: {e}")
        if samples_seen == 0:
            print(f"[EWC] No samples processed, skipping this task")
            self.task_count += 1
            self.net.train()
            return
    
    if samples_seen == 0:
        print(f"[EWC] No samples processed, skipping Fisher")
        self.task_count += 1
        self.net.train()
        return
    
    # Normalize
    for name in new_fisher:
        new_fisher[name] /= samples_seen
    
    # ========== Clip & Normalize ==========
    max_fisher = 10.0
    for name in new_fisher:
        new_fisher[name] = torch.clamp(new_fisher[name], max=max_fisher)
        fisher_max = new_fisher[name].max()
        if fisher_max > 1e-8:
            new_fisher[name] = new_fisher[name] / fisher_max
    
    # ========== Accumulate (Online EWC) ==========
    if not self.fisher:
        self.fisher = new_fisher
        print(f"[EWC] ✓ Initialized Fisher (Task {self.task_count + 1})")
    else:
        alpha = 1.0 / (self.task_count + 1)
        for name in self.fisher:
            self.fisher[name] = (1 - alpha) * self.fisher[name] + alpha * new_fisher[name]
        print(f"[EWC] ✓ Accumulated Fisher (Task {self.task_count + 1})")
    
    # Store optimal parameters
    self.old_params = {}
    for name, param in self.net.named_parameters():
        if param.requires_grad:
            self.old_params[name] = param.data.clone()
    
    # Stats
    total_fisher = sum(f.sum().item() for f in self.fisher.values())
    print(f"[EWC] Fisher stats: Total={total_fisher:.4f}, Tasks={self.task_count + 1}, Samples={samples_seen}")
    
    self.task_count += 1
    self.net.train()