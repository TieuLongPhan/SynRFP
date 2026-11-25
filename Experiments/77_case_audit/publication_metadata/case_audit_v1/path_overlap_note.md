# Why the Path r2 ceiling ranges overlap

**The large gain comes from mixed-label Path D classes.** Adding U across
the eight endpoints with a negative published MLP context effect yields
5,837 additional majority-correct rows. Mixed-label Path D classes account
for all 5,837. Classes that also combine distinct WL r2 D classes account for
5,836. Singletons contribute zero because their D majority is already correct.

SynTemp R2 accounts for 5,338 of those rows. Path r2 D gives 12,246 classes
and WL r2 D gives 35,569. Of the Path classes, 4,105 span multiple WL classes
and cover 35,024 of 43,439 encoded reactions (80.63%). These merger classes
have 84.76% row-weighted majority-label purity. Adding U resolves the
majority-label errors in 739 mixed-label classes, raising the empirical
ceiling from 87.71% to 100% (+12.29 percentage points). Refining the same Path
classes by WL D alone recovers 5,297 of the 5,338 majority-correct rows.
The published mean held-out context effect across model settings is −0.0109
MCC for SynTemp R2.

The Path r2 ceiling gains range from 0.27 to 16.62 percentage points across
the six positive-MCC endpoints and from 0.03 to 12.29 across the eight
negative-MCC endpoints. The second-largest gain in the negative-MCC group
is 0.54 points at Mech L2.

Within each Path D class, the gain is the sum of majority counts over D+U
subgroups minus the original majority count. A merger is a Path D class
that spans two or more WL D keys. Path and WL partitions can split across
each other. This audit uses the frozen Request 74 encoded cohort and exact
counts. The majority-label ceiling measures representational capacity on
that cohort.

The summary gives both reaction-weighted purity and the unweighted mean
purity across merger classes. [All 14 endpoints](path_overlap_summary.csv)
