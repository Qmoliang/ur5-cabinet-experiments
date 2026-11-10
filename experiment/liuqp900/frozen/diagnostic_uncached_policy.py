"""Single-factor diagnostic: rebuild continuation witnesses from the full map on every solve."""
from continuation_certified_policy import LowDimPolicy as Previous
class LowDimPolicy(Previous):
 def continuation_witnesses(self,q):
  self.witness_anchor=None
  return super().continuation_witnesses(q)
 def summary(self):
  out=super().summary();out['diagnostic_intervention']='full witness refresh every controller solve; original objective and execution preserved';return out
