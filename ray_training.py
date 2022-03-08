from typing import Dict

import torch

import ray

@ray.remote
class TrainingWorker:
    def __init__(self, 
                 rank: int, 
                 model_config: Dict):
        self.rank = rank
        self.device = torch.device(model_config["device"])
        self.model = model_config["model"].to(self.device)
        self.loss_fn = model_config["loss_fn"]
        self.optimizer = model_config["optimizer"](self.model.parameters(), 
                                                   lr=0.0001) 

    def train(self, shard):
                   
        epochs_data = []
        for epoch, batch in enumerate(shard.iter_batches(batch_format="native",
                                                         batch_size=1000)):
            print(f"Training... worker: {self.rank}, epoch {epoch}")
            total_loss = 0
            mol_acc = 0
            atom_acc = 0
            for data in batch:
                data = data.to(self.device)
                pred = self.model(data)
                loss = self.loss_fn(pred, data.y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                atom_acc += (pred.argmax(1).eq(data.y)).sum().item()/data.x.size(0)
                total_loss += self.loss_fn(pred, data.y).item()
                mol_acc += int(pred.argmax(1).eq(data.y).all())
            
            epochs_data.append({"loss": total_loss/len(batch),
                                "mol_acc": mol_acc/len(batch),
                                "atom_acc": atom_acc/len(batch),
                                "worker": self.rank
                                })
            if epoch % 10 == 0:
                print(f"Checkpoint worker: {self.rank}, loss: {total_loss/len(batch)}, acc: {mol_acc/len(batch)}")
                #torch.save(self.model.state_dict(), "ckp/weights_{}_{}.pt".format(self.rank, epoch))
        
        return epochs_data
        
                
    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)