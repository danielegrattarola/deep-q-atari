import os, time


class Logger:
	def __init__(self, debug = False):
		self.debug = debug
		if self.debug: return

		output_folder = './output/'
		run_folder = 'run%Y%m%d-%H%M%S/'
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)
		self.path = ''.join([output_folder, time.strftime(run_folder)])
		if not os.path.exists(self.path):
			os.makedirs(self.path)

		self.save_file = self.path + 'model.h5'

	def log(self, data):
		if self.debug:
			print data
			return
		try:
			logfile = open(self.path + 'log.txt', 'a')
		except IOError:
			print 'Logger:log IO error while opening log file'
			return
		if type(data) is dict:
			for k in data:
				logfile.write(k + ': ' + str(data[k]) + '\n')
				print k + ': ' + str(data[k])
		if type(data) is tuple:
			logfile.write(data[0] + ': ' + str(data[1]) + '\n')
		if type(data) is str:
			logfile.write(data + '\n')
			print data

	def to_csv(self, filename, row):
		if self.debug: return
		try:
			file = open(self.path + filename, 'a')
		except IOError:
			print 'Logger:to_csv IO error while opening file'
			return
		string = ','.join([str(val) for val in row])
		string = string + '\n' if not string.endswith('\n') else ''
		file.write(string)
		file.close()

