

class EnumerateConstants:
    TASK_FAIL_OUT_OF_DDL = 0
    TASK_FAIL_OUT_OF_TTI = 1
    TASK_FAIL_OUT_OF_NODE = 2
    
    CONSENSUS_POW = 11
    CONSENSUS_POS = 12

    @staticmethod
    def getDescByCode(code: int):
        """Get the description by the code.

        Args:
            code (int): The code.

        Returns:
            str: The description.
        """
        if code == EnumerateConstants.TASK_FAIL_OUT_OF_DDL:
            return "Task fails due to out of deadline."
        elif code == EnumerateConstants.TASK_FAIL_OUT_OF_TTI:
            return "Task fails due to out of transmission time interval."
        elif code == EnumerateConstants.TASK_FAIL_OUT_OF_NODE:
            return "Task fails due to out of node."
        elif code == EnumerateConstants.CONSENSUS_POW:
            return "Proof of work."
        elif code == EnumerateConstants.CONSENSUS_POS:
            return "Proof of stake."
        else:
            return "Unknown code."