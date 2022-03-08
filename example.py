from collections import OrderedDict
from pymongo import MongoClient

import torch
from torch.nn import CrossEntropyLoss
from torch_geometric.data import Data

import ray
from ray_dataset import MongoDatasource, create_large_shuffle_pipeline
from ray_training import TrainingWorker

from trf_models import TrfEdgeNet


NUM_EPOCHS = 10
NUM_GPUS = 4
DEVICE = "cuda"


def mongo_data_generator(skip, limit, collection):
    """
    Custom implementation of data generator from Mongo DB
    Args:
        skip (int): 
        limit (int): 
        collection (str): 

    Yields:
        torch_geometric.data.Data: torch data format for graphs
    """
    mc = MongoClient("stxnvme01")
    db = mc["MLshared"][collection]
    print("Picking data", skip, limit)
    for item in db.find({},
                        {"x": 1, "y": 1, "edge_attr": 1, "edge_index": 1}).skip(skip).limit(limit):
        yield Data(x=torch.tensor(item["x"], dtype=torch.float),
                   y=torch.tensor(item["y"], dtype=torch.long),
                   edge_attr=torch.tensor(item["edge_attr"], dtype=torch.float),
                   edge_index=torch.tensor(item["edge_index"], dtype=torch.long))
        
        

def run_for_collection(training_workers, splits): 
    result = ray.get([worker.train.remote(shard) 
                      for worker, shard in zip(training_workers, splits)])
    loss = 0.0
    mol_acc = 0.0
    atom_acc = 0.0
    for r in result:
        r = r[-1]
        loss += r["loss"]/len(result)
        mol_acc += r["mol_acc"]/len(result)
        atom_acc += r["atom_acc"]/len(result)
    
    print("Updating weights...")  
    weights = ray.get([worker.get_weights.remote() for worker in training_workers])
    updated_weights = OrderedDict((k, torch.zeros(v.size())) for k, v in weights[0].items())
    for w in weights:
        for k, v in w.items():
            updated_weights[k] += v.to(torch.device("cpu"))/NUM_EPOCHS
        
    return dict(loss=loss,
                mol_acc=mol_acc,
                atom_acc=atom_acc), \
           updated_weights 
    
        

if __name__ == "__main__":
    ray.init(num_gpus=4,
             num_cpus=16,
             dashboard_host='0.0.0.0')
    ray.data.set_progress_bars(True)
    
    mc = MongoClient("stxnvme01")
    
    model_config = {}
    model_config["model"] = TrfEdgeNet(139, 5, 492)
    model_config["loss_fn"] = CrossEntropyLoss()
    model_config["optimizer"] = torch.optim.Adam
    model_config["device"] = DEVICE
    
    training_workers = [TrainingWorker.options(num_gpus=1) \
                        .remote(rank, model_config) for rank in range(NUM_GPUS)]
    
    for db_num in range(1,2):
        collection_name = "AT_training_{}".format(db_num)
        data_count = mc["MLshared"][collection_name].estimated_document_count()
        print("\nCurrent collection:", collection_name, data_count)
        
        splits = create_large_shuffle_pipeline(data_count=data_count,
                                               data_generator=mongo_data_generator,
                                               collection=collection_name,
                                               parallel_tasks=NUM_GPUS, 
                                               num_epochs=NUM_EPOCHS,
                                               num_shards=NUM_GPUS)   
        
        accuracy, updated_weights = run_for_collection(training_workers,
                                                       splits)
        
        print("Current accuracy: loss - {}, mol acc. - {}, atom acc. - {}"
              .format(accuracy["loss"], accuracy["mol_acc"], accuracy["atom_acc"]))    
        
        weight_id = ray.put(updated_weights)      
        for trainer in training_workers:
            trainer.set_weights.remote(weight_id)  
            
    print("Yay!")
            
    # splits = create_large_shuffle_pipeline(data_count=data_count,
    #                                        data_generator=mongo_data_generator,
    #                                        collection=collection,
    #                                        parallel_tasks=NUM_GPUS, 
    #                                        num_epochs=NUM_EPOCHS,
    #                                        num_shards=1)     
    
    # for split in splits:
    #     for batch in split.iter_batches(batch_format="native"):
    #         print(batch[0:3])
        
        
