from bformat_segment import *
from make_folds import *

bformat_segment(['../FMS Selected/Albion Street/150626_354_st12.wav',
 '../FMS Selected/Albion Street/150626_354_st34.wav'], 'Albion-')

bformat_segment(['../FMS Selected/Dalby Forest/150625_344_st12.wav',
 '../FMS Selected/Dalby Forest/150625_344_st34.wav'], 'Dalby-')

bformat_segment(['../FMS Selected/Fox and Rabbit/150625_350_st12.wav',
 '../FMS Selected/Fox and Rabbit/150625_350_st34.wav'], 'Fox-')

bformat_segment(['../FMS Selected/Hole of Horcum/150625_348_st12.wav',
 '../FMS Selected/Hole of Horcum/150625_348_st34.wav'], 'Hole-')

bformat_segment(['../FMS Selected/Park Row/150626_356_st12.wav',
 '../FMS Selected/Park Row/150626_356_st34.wav'], 'Park-')

segment_dataset(3, ['Albion-','Dalby-','Fox-','Hole-','Park-'],
                    'eval_setup','audio')
