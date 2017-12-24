# encoding: UTF-8

"""
@author: hy
"""

import os

# preprocess
TEST_PROPORTION = 0.1   # all = train(0.9) + test(0.1)
VALIDATE_PROPORTION = 0.1   # train = train(0.9) + validate(0.1)
SCORE_WEIGHT = (0.5, 0.5)   # BLEU, GLEU

# path
PROJECT_PATH = os.path.dirname(os.path.split(os.path.realpath(__file__))[0])
DATA_PATH = PROJECT_PATH + os.sep + 'data'
CODE_PATH = PROJECT_PATH + os.sep + 'code'
SMT_RESULT_PATH = PROJECT_PATH + os.sep + 'SMTResult'
NMT_RESULT_PATH = PROJECT_PATH + os.sep + 'NMTResult'
TRAIN_DATA_PATH = DATA_PATH + '/preprocess/train.txt'
VALID_DATA_PATH = DATA_PATH + '/preprocess/validation.txt'
TEST_DATA_PATH = DATA_PATH + '/preprocess/test.txt'
LM_DATA_PATH = DATA_PATH + '/preprocess/LM.txt'
VOCAB_PATH = DATA_PATH + '/preprocess/vocab.txt'

# SMT
NEVER_OCCUR_CHAR = u'NEVER_OCCUR_CHAR'
NO_CHAR = u'NO_CHAR'
INTERACTIVE_MODE = 'interactive'
TESTFILE_MODE = 'testfile'


class SMTConfig(object):
	lmGramNum = 3               # gram number of language model
	smLambda = 0.1              # parameter of smothing method -- AddingLambda
	lmWeight = 0.3              # weight of language model
	beamSize = 10               # width of beam search

	def toStr(self):
		return 'lmn%d_sm%.3f_lmw%.3f_be%d' % (self.lmGramNum, self.smLambda, self.lmWeight, self.beamSize)

# NMT
TRAIN_MODE = 'TRAIN_MODE'
VALID_MODE = 'VALID_MODE'
INFER_MODE = 'TEST_MODE'

UNK = u'<unk>'
SOS = u'<sos>'
EOS = u'<eos>'

class NMTConfig(object):
	"""类变量, 当创建类对象时, 会将当前类变量拷贝一份赋给对象, 修改对象的变量并不改变类变量的值
	"""
	hiddenSize = 128    # unit number in RNN cell
	batchSize = 129     # batch size
	embedSize = 130     # batch
	keepRate = 0.5      # keep rate of dropout layer
	maxGradNorm = 5.0   # gradient clip
	lr = 0.001          # initial learning rate
	numLayers = 2       # layers of RNN
	vocabSize = 6000    # size of encoder vocabulary
	beamWidth=1         # width of beam search
	epochNum = 2000
	hyEpochNum = 5
	initScale = 0.1     # uniform init scale

	def toStr(self):
		return 'hs%d_bs%d_es%d_kr%.2f_mgn%.2f_lr%.4f_nl%d_vs%d_bw%d_ep%d' % (
			self.hiddenSize, self.batchSize, self.embedSize, self.keepRate,
			self.maxGradNorm, self.lr, self.numLayers, self.vocabSize, self.beamWidth, self.epochNum
		)

if __name__ == '__main__':
	config = NMTConfig()
	config.hiddenSize=1
	print config.hiddenSize, config.toStr()