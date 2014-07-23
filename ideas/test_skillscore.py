"""Test skill scores calculation for Jan streamflow forecast over the past 58 years.
The forecast consists of 5000 ensemble members.
"""
import unittest
import numpy as np
from fcvf import skillscore

class SkillScoreTestCase(unittest.TestCase):
    def setUp(self):
        self.yobs = np.loadtxt("yobs.csv", delimiter=",")
        self.ysim = np.loadtxt("ysim.csv", delimiter=",")
        self.yref = np.loadtxt("yref.csv", delimiter=",")

    def test_rmse(self):
        SSRMSE, RMSEsim, RMSEref = skillscore.rmse(self.yobs, self.ysim, self.yref)

        self.failUnlessAlmostEqual(SSRMSE,46.17383781)
        self.failUnlessAlmostEqual(RMSEsim,16.21174687)
        self.failUnlessAlmostEqual(RMSEref,30.11871218)

    def test_rmsep(self):
        SSRMSEP, RMSEPsim, RMSEPref = skillscore.rmsep(self.yobs, self.ysim, self.yref)

        self.failUnlessAlmostEqual(SSRMSEP,47.08819490)
        self.failUnlessAlmostEqual(RMSEPsim,0.14024542)
        self.failUnlessAlmostEqual(RMSEPref,0.26505507)

    def test_crps(self):
        SSCRPS, CRPSsim, CRPSref = skillscore.crps(self.yobs, self.ysim, self.yref)

        self.failUnlessAlmostEqual(SSCRPS,47.997812480261835)
        self.failUnlessAlmostEqual(CRPSsim,7.1688396864064821)
        self.failUnlessAlmostEqual(CRPSref,13.7856502)

    def test_2dim_yref(self):
        SSCRPS, CRPSsim, CRPSref = skillscore.crps(self.yobs, self.ysim, self.yref)
        yref2d = [self.yref]*len(self.yobs)
        SSCRPS2d, CRPSsim2d, CRPSref2d = skillscore.crps(self.yobs, self.ysim, yref2d)
        self.failUnlessAlmostEqual(SSCRPS,SSCRPS2d)
        self.failUnlessAlmostEqual(CRPSsim,CRPSsim2d)
        self.failUnlessAlmostEqual(CRPSref,CRPSref2d)

    def test_obs_nan_check_with_rmsep(self):
        self.yobs[5] = np.nan
        self.assertRaises(ValueError, skillscore.rmsep, self.yobs, self.ysim, self.yref)

    def test_ref_nan_check_with_rmsep(self):
        self.yref[8] = np.nan
        self.assertRaises(ValueError, skillscore.rmsep, self.yobs, self.ysim, self.yref)

    def test_sim_nan_check_with_rmsep(self):
        self.ysim[2] = np.nan
        self.assertRaises(ValueError, skillscore.rmsep, self.yobs, self.ysim, self.yref)

    def test_sim_len_check_with_rmsep(self):
        ysim = np.append(self.ysim, 1)
        self.assertRaises(ValueError, skillscore.rmsep, self.yobs, ysim, self.yref)

if __name__ == "__main__":
    unittest.main()

