def LowDimPolicy(ctx,job,event):
 if job['arm']=='A':
  from baseline_policy import LowDimPolicy as Policy
 elif job['arm']=='B':
  from configuration_recurrence_policy import LowDimPolicy as Policy
 else:raise ValueError(job['arm'])
 return Policy(ctx,job,event)
