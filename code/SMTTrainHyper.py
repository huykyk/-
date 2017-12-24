# encoding: UTF-8

"""
@author: hy
"""

from SMT import SMTModel
from multiprocessing import Pool, cpu_count
from SMTConfig import *
import itertools
from Common import *


def runTest(params):
	indexStr, data, lmGramNum, smoothingLambda, lmWeight, beamSize = params
	print 'running %s |poemSMT_lmn%d_sm%.3f_lmw%.3f_be%d...' % (indexStr, lmGramNum, smoothingLambda, lmWeight, beamSize)
	model = SMTModel(lmGramNum, smoothingLambda, lmWeight, beamSize)
	model.data = data
	model.train(data['train'], data['lm'])
	ret = model.test(data['validation'])
	print indexStr, 'end:', ret[2]
	return ret


def saveResult(resultList, path):
	with open(path, 'w') as f:
		for result in resultList:
			f.write(result[2] + '\n')


if __name__ == '__main__':
	data = readData(
		trainPath=DATA_PATH + '/preprocess/train.txt',
		validationPath=DATA_PATH + '/preprocess/validation.txt',
		testPath=DATA_PATH + '/preprocess/test.txt',
		lmPath=DATA_PATH + '/preprocess/LM.txt'
	)
	lmnList = [2, 3]
	smList = [0.05, 0.1, 0.5, 1.0]
	lmwList = [0.0, 0.3, 0.5, 0.8]
	beList = [10]

	param = list(itertools.product(lmnList, smList, lmwList, beList))
	param = [('%d/%d'%(i+1,len(param)), data)+param[i] for i in range(len(param))]
	print 'grid number:', len(param)

	pool = Pool(processes=4)
	result = pool.map(runTest, param)
	pool.close()
	pool.join()

	saveResult(sorted(result, reverse=True), SMT_RESULT_PATH + '/bleu_first.txt')
	saveResult(sorted(result, key=lambda item: (item[1], item[0], item[2]), reverse=True), SMT_RESULT_PATH + '/gleu_first.txt')
