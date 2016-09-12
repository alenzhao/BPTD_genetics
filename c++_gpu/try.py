




if __name__ == "__main__":



	##====
	file = open("file.txt", 'r')

	hashtable_gpu = {}

	while 1:
		line = (file.readline()).strip()
		if not line:
			break

		line = line.split(' ')
		if len(line) < 5:
			continue

		id = line[0]
		if id[:4] == "gpu_" or id[3:6] == "gpu":
			host = id.split('@')[1]
			hashtable_gpu[host] = id
		
	file.close()


	print len(hashtable_gpu)



	##====
	file = open("file.txt", 'r')

	while 1:
		line = (file.readline()).strip()
		if not line:
			break

		line = line.split(' ')
		if len(line) < 5:
			continue

		id = line[0]
		if id[:4] == "gpu_" or id[3:6] == "gpu":
			continue

		host = id.split('@')[1]
		if host in hashtable_gpu:
			print hashtable_gpu[host]
			print line

		

	file.close()


