class EnumerateConstants:
    TASK_FAIL_OUT_OF_DDL = 0
    TASK_FAIL_OUT_OF_TTI = 1
    TASK_FAIL_OUT_OF_NODE = 2
    TASK_FAIL_PARENT_FAILED = 3
    TASK_FAIL_MALICIOUS_RESULT = 4  # 🔧 新增：恶意结果导致的任务失败

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
            return "Task fails due to transmission timeout."
        elif code == EnumerateConstants.TASK_FAIL_OUT_OF_NODE:
            return "Task fails due to out of node."
        elif code == EnumerateConstants.TASK_FAIL_PARENT_FAILED:
            return "Task fails due to parent task failed."
        elif code == EnumerateConstants.TASK_FAIL_MALICIOUS_RESULT:
            return "Task fails due to malicious result detected."
        elif code == EnumerateConstants.CONSENSUS_POW:
            return "Proof of work."
        elif code == EnumerateConstants.CONSENSUS_POS:
            return "Proof of stake."
        else:
            return "Unknown code."


class NodeTypeEnum:
    CLOUD_SERVER = 0
    RSU = 1
    VEHICLE = 2
    UAV = 3
