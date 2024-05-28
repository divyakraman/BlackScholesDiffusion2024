'''
# Set 1

objects = ['a rock', 'a coffee mug', 'a cute dog', 'a pink cat', 'a teddy bear', 'a robot', 'an alien', 'an avocado', 'a cute raccoon', 'a corgi dog', 'a parrot', 'a car', 'a squirrel', 'a cute rabbit', 'pizza', 'muffin', 'icecream']

f = open('set1.txt', 'w')
counter = 0

for i in range(len(objects)):
	for j in range(i+1, len(objects)):
		counter = counter + 1
		f.write('prompt' + str(counter) + '\t')
		prompt0 = objects[i] + ' in the shape of ' + objects[j]
		prompt1 = objects[j] + ' in the shape of ' + objects[i]
		prompts_list = '[' + prompt0 + ',' + prompt1 + ',' + objects[i] + ',' + objects[j] + ']'
		f.write(prompts_list)
		f.write('\n')

f.close()

'''

'''
# Set 2

objects1 = ['an apple', 'a banana', 'a cute cat', 'a corgi dog', 'a muffin']
objects2 = ['a basket', 'a teapot']
objects3 = ['a table', 'a bed', 'a carpet']

f = open('set2.txt', 'w')
counter = 0

for i in range(len(objects2)):
	for j in range(len(objects1)):
		for k in range(j+1, len(objects1)):
			counter = counter + 1
			f.write('prompt' + str(counter) + '\t')
			prompt0 = objects2[i] + ' with ' + objects1[j] + ' in the shape of ' + objects1[k]
			prompt1 = objects2[i] + ' with ' + objects1[k] + ' in the shape of ' + objects1[j]
			prompt2 = objects2[i] + ' with ' + objects1[j]
			prompt3 = objects2[i] + ' with ' + objects1[k]
			
			prompts_list = '[' + prompt0 + ',' + prompt1 + ',' + prompt2 + ',' + prompt3 + ']'
			f.write(prompts_list)
			f.write('\n')

for i in range(len(objects1)):
	for j in range(len(objects2)):
		for k in range(j+1, len(objects2)):
			counter = counter + 1
			f.write('prompt' + str(counter) + '\t')
			prompt0 = objects2[j] + ' in the shape of ' + objects2[k] + ' with ' + objects1[i]
			prompt1 = objects2[k] + ' in the shape of ' + objects2[k] + ' with ' + objects1[i]
			prompt2 = objects2[j] + ' with ' + objects1[i]
			prompt3 = objects2[k] + ' with ' + objects1[i]
			
			prompts_list = '[' + prompt0 + ',' + prompt1 + ',' + prompt2 + ',' + prompt3 + ']'
			f.write(prompts_list)
			f.write('\n')

for i in range(len(objects3)):
	for j in range(len(objects1)):
		for k in range(j+1, len(objects1)):
			counter = counter + 1
			f.write('prompt' + str(counter) + '\t')
			prompt0 = objects1[j] + ' in the shape of ' + objects1[k] + ' on ' + objects3[i]  
			prompt1 = objects1[k] + ' in the shape of ' + objects1[j] + ' on ' + objects3[i]
			prompt2 = objects1[j] + ' on ' + objects3[i]
			prompt3 = objects1[k] + ' on ' + objects3[i]
			
			prompts_list = '[' + prompt0 + ',' + prompt1 + ',' + prompt2 + ',' + prompt3 + ']'
			f.write(prompts_list)
			f.write('\n')

for i in range(len(objects1)):
	for j in range(len(objects3)):
		for k in range(j+1, len(objects3)):
			counter = counter + 1
			f.write('prompt' + str(counter) + '\t')
			prompt0 = objects1[i] + ' on ' + objects3[j] + ' in the shape of ' + objects3[k] 
			prompt1 = objects1[i] + ' on ' + objects3[k] + ' in the shape of ' + objects3[j]
			prompt2 = objects1[i] + ' on ' + objects3[j]
			prompt3 = objects1[i] + ' on ' + objects3[k]
			
			prompts_list = '[' + prompt0 + ',' + prompt1 + ',' + prompt2 + ',' + prompt3 + ']'
			f.write(prompts_list)
			f.write('\n')

f.close()

'''

'''
# Set 3

objects1 = ['a kangaroo', 'a cute cat', 'a corgi dog', 'a parrot', 'a teddy bear', 'a penguin']
backgrounds = ['walking in Times Square', 'skiing in Times Square', 'walking in a beautiful garden', 'surfing on the beach', 'eating watermelon on the beach', 'sitting on a sofa on the beach', 'watching northern lights', 'watching sunset at a beach', 'admiring the opera house in Sydney', 'sleeping in a cozy bedroom', 'admiring a beautiful waterfall in a forest', 'walking in a cherry blossom garden', 'walking in a colorful autumn forest', 'flying in the sky at sunset']

f = open('set3.txt', 'w')
counter = 0

for i in range(len(backgrounds)):
	for j in range(len(objects1)):
		for k in range(j+1, len(objects1)):
			counter = counter + 1
			f.write('prompt' + str(counter) + '\t')
			prompt0 = objects1[j] + ' in the shape of ' + objects1[k] + ' ' + backgrounds[i]
			prompt1 = objects1[k] + ' in the shape of ' + objects1[j] + ' ' + backgrounds[i]
			prompt2 = objects1[j] + ' ' + backgrounds[i]
			prompt3 = objects1[k] + ' ' + backgrounds[i]
			
			prompts_list = '[' + prompt0 + ',' + prompt1 + ',' + prompt2 + ',' + prompt3 + ']'
			f.write(prompts_list)
			f.write('\n')

'''

# Set 4

objects1 = ['a kangaroo', 'a cute cat', 'a corgi dog', 'a parrot', 'a teddy bear', 'a penguin']
backgrounds = [' walking in Times Square', ' watching the sunset', ' in van gogh style', ' in oil painting style', ' walking in a tulip garden', ' watching the northern lights']

f = open('set4.txt', 'w')
counter = 0

for i in range(len(objects1)):
	for j in range(len(backgrounds)):
		for k in range(j+1, len(backgrounds)):
			counter = counter + 1
			f.write('prompt' + str(counter) + '\t')
			prompt0 = objects1[i] + backgrounds[j] + ';' + backgrounds[k]
			prompt1 = objects1[i] + backgrounds[k] + ';' + backgrounds[j]
			prompt2 = objects1[i] + backgrounds[j]
			prompt3 = objects1[i] + backgrounds[k]
			
			prompts_list = '[' + prompt0 + ',' + prompt1 + ',' + prompt2 + ',' + prompt3 + ']'
			f.write(prompts_list)
			f.write('\n')
f.close()

