"""
refernce: https://github.com/nvidia/mig-parted
"""


class MIGController:
    """
    In charge of mig partition and recovery
    """
    def __init__(self):
        pass

    def partition(self):
        """
        takes in an assigned partition(gpiID with mig partition),
        if success, returns a data structure contains the deviceIds of all created mig device.
        Return example:
            mig-devices:
                1g.5gb: 'MIG-29f07255-51c0-5691-82c4-cbc57760ff63'
                2g.10gb: 'MIG-ad654d5e-db91-50e1-bcf4-1e9c24f825ad'
                3g.20gb: 'MIG-ea08ed7b-a485-5967-9c78-2fa6e548c43a'
        """
        pass

    def recover(self):
        """
        stop all cuda processes on img devices and then delete all mig devices.
        """
        pass

