import unittest

from migperf.controller import MIGController

unittest.TestLoader.sortTestMethodsUsing = None


class MIGControllerEnableDisableTest(unittest.TestCase):
    mig_controller = None

    @classmethod
    def setUpClass(cls):
        cls.mig_controller = MIGController()

    @classmethod
    def tearDownClass(cls):
        cls.mig_controller.disable_mig()
        cls.mig_controller = None

    def test_enable_mig_controller_single_gpu(self):
        self.mig_controller.enable_mig(gpu_id=0)
        mig_status = self.mig_controller.check_mig_status(gpu_id=0)
        self.assertEqual(tuple(mig_status), (True, True))
        self.mig_controller.enable_mig(gpu_id=1)
        mig_status = self.mig_controller.check_mig_status(gpu_id=1)
        self.assertEqual(tuple(mig_status), (True, True))

    def test_disable_mig_controller_single_gpu(self):
        self.mig_controller.disable_mig(gpu_id=0)
        mig_status = self.mig_controller.check_mig_status(gpu_id=0)
        self.assertEqual(tuple(mig_status), (False, False))
        self.mig_controller.disable_mig(gpu_id=1)
        mig_status = self.mig_controller.check_mig_status(gpu_id=1)
        self.assertEqual(tuple(mig_status), (False, False))

    def test_enable_mig_controller_all(self):
        self.mig_controller.enable_mig()
        mig_status_list = self.mig_controller.check_mig_status()
        for mig_status in mig_status_list:
            self.assertEqual(tuple(mig_status), (True, True))

    def test_disable_mig_controller_all(self):
        self.mig_controller.disable_mig()
        mig_status_list = self.mig_controller.check_mig_status()
        for mig_status in mig_status_list:
            self.assertEqual(tuple(mig_status), (False, False))


class MIGControllerMIGDeviceSingleGPUTest(unittest.TestCase):
    mig_controller = None
    gpu_id = 0

    @classmethod
    def setUpClass(cls):
        cls.mig_controller = MIGController()
        cls.mig_controller.enable_mig(gpu_id=0)

    @classmethod
    def tearDownClass(cls):
        cls.mig_controller.disable_mig(gpu_id=0)
        cls.mig_controller = None

    def test_walk_through(self):
        gi_instances = self.mig_controller.create_gpu_instance(gi_profiles='1g.10gb,1g.10gb', gpu_id=0)
        print(gi_instances)
        print(self.mig_controller.check_gpu_instance_status(gpu_id=0))
        ci_instances = self.mig_controller.create_compute_instance(ci_profiles='1g.10gb')
        print(ci_instances)
        print(self.mig_controller.check_compute_instance_status(gpu_id=0))
        self.mig_controller.destroy_compute_instance(gpu_id=0)
        self.mig_controller.destroy_gpu_instance(gpu_id=0)


if __name__ == '__main__':
    unittest.main()
