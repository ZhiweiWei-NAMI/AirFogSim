import hashlib
from ..enum_const import EnumerateConstants
class Block():
    """ Block is the class that represents the block in the blockchain. 
    """
    def __init__(self, block_timestamp, block_transactions, block_previous_hash):
        """The constructor of the Block class.

        Args:
            block_timestamp (str): The timestamp of the block generated.
            block_transactions (list): The transaction information of the block.
            block_previous_hash (float): The            of the task.
            
            Each block is "chained" to its previous one by calling its unique hash
        """
        self.block_timestamp = block_timestamp # 产生的时间戳
        self.block_transactions = block_transactions
        self.block_previous_hash = block_previous_hash
        self.block_index = 0
        self.block_miner = None
        self.block_hash = self.get_hash()

    def getTransactionNum(self):
        """Get the number of the transactions in the block.

        Returns:
            int: The number.
        """
        return len(self.block_transactions)

    def get_hash(self):
        """Creates the unique hash for the block using sha256.

        Returns:
            float: The hash value of the new block.
        """
        
        sha = hashlib.sha256()
        sha.update((str(self.block_timestamp) + str(self.block_transactions) + str(self.block_previous_hash)).encode('utf-8'))
        return sha.hexdigest()
    
class Blockchain():
    """ Blockchain is the class that represents the blockchain. 
    """
    def __init__(self, cur_time, mine_time_threshold = 10, transaction_threshold = 10, consensus = EnumerateConstants.CONSENSUS_POS):
        """The constructor of the Block class.

        Args:
            cur_time (str): The current time.
            mine_time_threshold (int): 
            transaction_threshold (int): 
            consensus(str): The consensus algorithm used by the blockchain.
            
        """
        self.all_transactions = []
        self.last_update_time = cur_time
        self.mine_time_threshold = mine_time_threshold
        self.transaction_threshold = transaction_threshold
        self.consensus_type = consensus
        self.chain = [self.create_genesis_block()]
        print("区块链初始化完成, 共识机制为: ", EnumerateConstants.getDescByCode(consensus))
        self.to_mine_blocks = []
    
    @property
    def length(self):
        """Get the length of the blockchian.

        Returns:
            int: The length.
        """
        return len(self.chain)
    
    @property
    def transaction_num(self):
        """Get the number of the transactions in the blockchian.

        Returns:
            int: The number.
        """
        return len(self.all_transactions)
    
    def getBlockByIndex(self, idx):
        """Get the block by the index.

        Args:
            idx (int): The index of the block.

        Returns:
            Block: The block.
        """
        return self.chain[idx]

    def create_genesis_block(self):
        """Creates the genesis block of the blockchain.

        Returns:
            Block: The new block.
        """
        return Block(self.last_update_time, [], "0")

   

    def setConsensus(self, consensus):
        """Set the consensus of the blockchain.

        Args:
            consensus (int): The consensus.

        Returns:
            None.
        """
        self.consensus_type = consensus

    def getConsensus(self):
        """Get the consensus of the blockchain.

        Returns:
            int: The consensus.
        """
        return self.consensus_type
    def getMineTimeThreshold(self):
        """Get the threshold of the mine time.

        Returns:
            int: The threshold.
        """
        return self.mine_time_threshold
    def setMineTimeThreshold(self, threshold):
        """Set the threshold of the mine time.

        Args:
            threshold (int): The threshold.

        Returns:
            None.
        """
        self.mine_time_threshold = threshold