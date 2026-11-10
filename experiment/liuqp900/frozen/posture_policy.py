"""Rank-independent two-feature posture objective; original maximin constraints remain."""
import numpy as np
from legacy_goal_plane_memory_policy import LowDimPolicy as Previous
from goal_entry_policy import LowDimPolicy as Entry
class LowDimPolicy(Previous):
 def _terms(self,c,s):
  if self.mode!='CERT_PREP':return super()._terms(c,s)
  # Two posture features remain a secondary task. Their compatibility is
  # enforced by the maximin inequalities, not nulling ALL witness rows.
  return Entry._terms(self,c,s)
 def summary(self):
  out=super().summary();out.update(t0_revision='lexicographic_two_feature_posture',posture_dimension=2,posture_disabled_by_full_witness_rank=False);return out
