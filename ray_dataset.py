from email import generator
from typing import List, Optional

import torch_geometric

import ray
from ray.data import read_datasource
from ray.data.datasource import Datasource, ReadTask
from ray.data.block import Block, T, BlockMetadata
from ray.data.dataset_pipeline import DatasetPipeline


class MongoDatasource(Datasource):
    """
    Class implementing ray.data.datasource.Datasource interface 
    Main modification - introduction of `mongo_data_generator` method that reads data directly from Mongo DB
    to be able to read different mongo collections with various data schema, implement your own generator and pass upon class initialization
    Args:
        data_count: total number of data in the dataset
        mongo_data_generator: generator function picking data from wherever needed
        collection: (optional) for Mongo db
        
    """
    def __init__(self,
                 data_count: int,
                 mongo_data_generator: generator,
                 collection: Optional[str]
                 ) -> None:
        super(MongoDatasource).__init__()
        self.data_count = data_count
        self.collection = collection
        self.mongo_data_generator = mongo_data_generator
        
    def prepare_read(self, 
                     parallelism: int
                     ) -> List["ReadTask[T]"]:
        read_tasks: List[ReadTask] = []
        block_size = max(1, self.data_count // parallelism)
        
        def make_block(skip: int,
                       limit: int) -> Block:
            dataset = [d for d in self.mongo_data_generator(skip, limit, self.collection)]
            return dataset    
                    
        i = 0
        while i < self.data_count:
            print("Current batch:", i)
            count = min(block_size, self.data_count - i)
            schema=torch_geometric.data.Data
            meta = BlockMetadata(
                num_rows=count,
                size_bytes=8 * count,
                schema=schema,
                input_files=None,
                exec_stats=None)
            
            read_tasks.append(
                ReadTask(lambda i=i, count=count: [make_block(i, count)], 
                         meta)
                )
            i += block_size
            
        return read_tasks
    
    
def create_large_shuffle_pipeline(data_count, 
                                  data_generator,
                                  collection,
                                  parallel_tasks: int, 
                                  num_epochs: int,
                                  num_shards: int) -> List[DatasetPipeline]:
    """
    Create pipeline based on data generated from Mongo DB to feed the ray trainer 
    """
    # _spread_resource_prefix is used to ensure tasks are evenly spread to all
    # CPU nodes.
    return read_datasource(MongoDatasource(data_count, data_generator, collection), 
                           parallelism=parallel_tasks,
                           _spread_resource_prefix="node:") \
                                .repeat(num_epochs) \
                                .random_shuffle_each_window(_spread_resource_prefix="node:") \
                                .split(num_shards, equal=True)