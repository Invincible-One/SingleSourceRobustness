import torch
from result_collect import ResCollector

def run_epoch(
        dataloader,
        network,
        loss_computer,
        optimizer,
        train,
        collect_res=False,
        epoch=None,
        total_data_points=None,
        args=None,
        device=torch.device('cuda'),
        ):
    if train:
        network.train()
    else:
        network.eval()
        assert collect_res == False
    
    ##collect results
    if collect_res:
        assert epoch is not None
        assert total_data_points is not None
        assert args is not None
        res_collector = ResCollector(epoch, total_data_points=total_data_points, args=args)
    else:
        res_collector = None
    
    loss_computer.reset_stats()

    with torch.set_grad_enabled(train):
        for batch_idx, batch in enumerate(dataloader):
            batch = tuple(e.to(device) for e in batch)
            X, y, g, data_idx = batch[0], batch[1], batch[2], batch[3]
            if train:
                optimizer.zero_grad()
            
            output = network(X)
            loss_v = loss_computer.loss(output, y)
            
            if train:
                loss_v.backward()
                optimizer.step()

            ##collect results
            if res_collector is not None:
                res_collector.update_res(output=output, y=y, data_idx=data_idx)
            
            loss_computer.update_stats(output.detach(), y, g)
    
    if res_collector is not None:
        res_collector.save_res()
    
    return loss_computer.display(train)
