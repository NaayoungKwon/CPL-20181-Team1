import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from enum import Enum
import detect_line as dl
import statistics
import math
from util import *

# If executed directly
def main():
	original_img = cv2.imread('../preprocessing/book_sample_0_crop_0.png')
	grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
	(_, im_bw) = cv2.threshold(grayscale_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	(x, y) = im_bw.shape

	para = Paragraph(img=im_bw)
	essay = dl.output_line_imgs(para)
	save_essay(essay)
	get_graphs(para)

# Guesses one letter size
def guess_letter_size(wlist, height):
	wlist.sort()
	l = int(height * 0.8)
	h = int(height * 1.2)
	bucket = [0]*(h - l)

	length = len(wlist)
	if length == 0: return 0
	err = 2
	j = 0
	for i in range (l, h):
		if (wlist[j] > i + err):
			break
		flag = 0
		for k in range (j, length):
			if wlist[k] > i - err and flag == 0:
				j = k
				flag = 1
	
			if wlist[k] >= i - err and wlist[k] <= i + err:
				bucket[i - l] = bucket[i - l] + 1
			elif wlist[k] < i - err:
				continue
			else:
				break
	m, index = -1, -1
	for i in range (l, h):
		if (m < bucket[i - l]):
			m, index = bucket[i - l], i

	return index

# Returns all the candidate letter points as a list of pairs and max width of letter
# Return value:
#		( maximum width of a letter in the line,
#				[ word1, word2, word3, ... , wordN ] )
# wordN ::= [ letter1, letter2, ... , letterN ]
# letterN ::= ( starting index, ending index )
def fst_pass(line):
	candidate = []
	word = []
	start_letter, end_letter = -1, -1
	wlist = []
	
	blank_thd = 5100 # Fix this to adjust blank column threshold

	(height, length) = line.shape
	for i in range (0, length):
		sum = sumup_col(line, i)
		if sum > blank_thd and start_letter == -1:
			if i == 0:
				start_letter = i
			else:
				start_letter = i - 1
			if start_letter - end_letter > 0.2 * height: # Fix this to adjust space threshold
				if word != []:
					candidate.append(word)
					word = []
		elif sum > blank_thd or (sum <= blank_thd and start_letter == -1):
			continue
		elif sum <= blank_thd:
			end_letter = i
			width = end_letter - start_letter
			if (width >= 4): # Fix this to adjust noise threshold
				wlist.append(width)
				word.append((start_letter, end_letter))
			start_letter = -1

	if start_letter != -1:
		width = length - 1 - start_letter
		wlist.append(width)
		word.append((start_letter, length - 1))
	if word != []:
		candidate.append(word)
	return (guess_letter_size(wlist, height), candidate)

# Calculates the difference of end and start pts of each letter, returns a list of widths
def calc_width(word):
	width = []
	for i in word:
		width.append(i[1] - i[0])
	return width

FIND_SPLIT_STRICT = 40 # 클수록 letter_size와 비슷한 지점을 찾는다
def find_split_pts(index, num_letter, size, img):
	(h, w) = img.shape
	i = 0
	pts = []

	min_pt = round(size * 0.6)
	max_pt = round(size * 1.4)

	while True:
		pt = i
		m = float("inf")
		for j in range (i + min_pt, min(w, i + max_pt)):
			s = sumup_col(img, j) + ((j - i - size) ** 2) * FIND_SPLIT_STRICT * h
			if s <= m:
				pt = j
				m = s
		pts.append((index + i, index + pt))
		if pt == i or pt+size * 1.5 >= w:
			break
		i = pt

	return pts

# Merges and splits letters considering the context
# Return value:
#		[ word1, word2, word3, ... , wordN ]
# wordN ::= [ letter1, letter2, ... , letterN ]
# letterN ::= ( starting index, ending index )
def snd_pass(line, size, candidate, merge_thdl, merge_thdh, flag):
	snd_candidate = []
	(height, _) = line.shape
	for w in candidate:
		word = list(w)
		snd_word = []
		width = calc_width(word)
		length = len(width)
		for i in range (0, length - 1):
			if flag == 0:
				if size * merge_thdl <= word[i + 1][1] - word[i][0] <= height * merge_thdh and width[i] >= width[i + 1]:
					word[i + 1] = (word[i][0], word[i + 1][1])
					width[i + 1] = word[i + 1][1] - word[i][0]
				elif width[i] >= height and width[i + 1] < size / 2:
					word[i + 1] = (word[i][0], word[i + 1][1])
					width[i + 1] = word[i + 1][1] - word[i][0]
				elif width[i] < size * 0.7 and width[i + 1] > height:
					word[i + 1] = (word[i][0], word[i + 1][1])
					width[i + 1] = word[i + 1][1] - word[i][0]
				elif width[i + 1] > height:
					word[i + 1] = (word[i][0], word[i + 1][1])
					width[i + 1] = word[i + 1][1] - word[i][0]
				else:
					snd_word.append(word[i])
			else:
				word[i + 1] = (word[i][0], word[i + 1][1])
				width[i + 1] = word[i + 1][1] - word[i][0]
		snd_word.append(word[length - 1])
		snd_candidate.append(snd_word)
	return snd_candidate

# split merged letters
def thd_pass(line, size, candidate):
	thd_candidate = []
	(height, _) = line.shape
	for word in candidate:
		thd_word = []
		width = calc_width(word)
		length = len(width)
		for i in range (0, length):
			if width[i] > height:
				num_letter = round(width[i] / size)
				pts = find_split_pts(word[i][0], num_letter, size, line[0:,word[i][0]:word[i][1]])
				l = len(pts)
				for j in range (0, l):
					thd_word.append(pts[j])
				if l != 0:
					thd_word.append((pts[l - 1][1], word[i][1]))
				else:
					thd_word.append(word[i])
			else:
				thd_word.append(word[i])
		thd_candidate.append(thd_word)
	return thd_candidate

def proc_line(line_img, letter_size, fst_c, merge_thdl, merge_thdh, flag):
	if letter_size < 2 : return []
	candidate = snd_pass(line_img, letter_size, fst_c, merge_thdl, merge_thdh, flag)
	candidate = thd_pass(line_img, letter_size, candidate)
	return candidate
'''
# Given a paragraph, processes 2 times and returns the candidate word points
def proc_paragraph(para):
	fst_candidate = []
	snd_candidate1 = []
	snd_candidate2 = []
	snd_candidate3 = []
	snd_candidate4 = []
	wlist = []
	for line in para.lines:
		(m, c) = fst_pass(line.img)
		fst_candidate.append(c)
		wlist.append(m)
	if len(wlist) == 0:
		return None
	letter_size = statistics.median(wlist)
	if letter_size < 1:
		return None
	for i, line in enumerate(para.lines):
		c1 = snd_pass(line.img, letter_size, fst_candidate[i], 0.6, 1.3, 0)
		c1 = thd_pass(line.img, letter_size, c1)
		snd_candidate1.append(c1)
		c2 = snd_pass(line.img, letter_size, fst_candidate[i], 0.8, 1.1, 0)
		c2 = thd_pass(line.img, letter_size, c2)
		snd_candidate2.append(c2)
		c3 = snd_pass(line.img, letter_size, fst_candidate[i], 0.7, 0.9, 0)
		c3 = thd_pass(line.img, letter_size, c3)
		snd_candidate3.append(c3)
		# Generous merge threshold
		c4 = snd_pass(line.img, letter_size, fst_candidate[i], None, None, 1)
		c4 = thd_pass(line.img, letter_size, c4)
		snd_candidate4.append(c4)
	return (snd_candidate1, snd_candidate2, snd_candidate3, snd_candidate4)
'''

# Implemented for testing; truncates line image to letters and saves letter images
# with word and letter information
def trunc_n_save_letter(para_num, line_num, pass_num, line, candidate):
	(x, y) = line.shape
	cnt_word = 0
	for word in candidate:
		cnt_letter = 0
		for letter in word:
			filepath = ('test/char_testing/t' + str(pass_num) + '/p' + str(para_num) + 'l' + str(line_num) +
					'w' + str(cnt_word) + 'l' + str(cnt_letter) + '.png')
			success = cv2.imwrite(filepath, line[0:x,letter[0]:letter[1]])
			if not success: print("save %s failed: check if folder exists" % filepath)
			cnt_letter = cnt_letter + 1
		cnt_word = cnt_word + 1

'''
# Implemented for testing; Given an essay, processes 2 times and saves letters as images
def save_essay(essay):
	for i, graph in enumerate(essay):
		(c1, c2, c3, c4) = proc_paragraph(graph)
		for j, line in enumerate(graph.lines):
			trunc_n_save_letter(i, j, 1, line.img, c1[j])
			trunc_n_save_letter(i, j, 2, line.img, c2[j])
			trunc_n_save_letter(i, j, 3, line.img, c3[j])
			trunc_n_save_letter(i, j, 4, line.img, c4[j])
'''

# Given four lists (candidate lists of different threshold),
# finds the indices for each list with the same end points
# returns them in a tuple along with the value of the end point
def find_first_common_pt(clists):
	idxes = [0 for clist in clists]
	lens = [len(clist) for clist in clists]
	end_list = [clist[0][1] for clist in clists]
	min_end = min(end_list)

	while True:
		abort = True
		for i in range(1, len(end_list)):
			if end_list[i-1] != end_list[i]:
				abort = False
				break
		if abort: break

		abort = False
		for i in range(len(end_list)):
			if end_list[i] == min_end:
				idxes[i] += 1
				if idxes[i] >= lens[i]:
					abort = True
					break
				end_list[i] = clists[i][idxes[i]][1]
		if abort: break
		min_end = min(end_list)
	
	return min_end, idxes

# Given the four candidates, each candidate having form of
# candidate ::= [ point1, point2, ... , pointN ]
# pointn ::= (start index, end index)
# returns list of characters
def compare_pass(candidates):
	char_list = []
	idxes = [0 for cand in candidates]
	lens = [len(cand) for cand in candidates]
	while True:
		abort = False
		for idx, l in zip(idxes, lens):
			if idx >= l:
				abort = True
				break
		if abort: break

		m, points = find_first_common_pt([cand[idx:] for cand, idx in zip(candidates, idxes)])
		char = Char((candidates[0][idxes[0]][0], m), CHARTYPE.CHAR) # Start point
		if max(points) != 0:
			for cand, idx, point in zip(candidates, idxes, points):
				curr = char
				for i in range (0, point + 1):
					child = curr.get_child(cand[idx + i])
					if child == None:
						a = Char(cand[idx + i], CHARTYPE.CHAR)
						curr.add_child(a)
						curr = a
					else:
						curr = child
		char_list.append(char)

		for i in range(len(idxes)):
			idxes[i] += points[i] + 1
	return char_list

'''
def argmin(items):
	min_idxs = []
	min_elm = float("inf")
	for idx, elm in enumerate(items):
		if min_elm > elm:
			min_idxs = [idx]
			min_elm = elm
		elif min_elm == elm:
			min_idxs.append(idx)
	return min_idxs, min_elm
'''

# Truncates line image to letters and constructs line classes
def get_char_list(line_img):
	(letter_size, fst_c) = fst_pass(line_img)

	p1 = proc_line(line_img, letter_size, fst_c, 0.6, 1.3, 0)
	p2 = proc_line(line_img, letter_size, fst_c, 0.8, 1.1, 0)
	p3 = proc_line(line_img, letter_size, fst_c, 0.7, 0.9, 0)
	#p4 = proc_line(line_img, letter_size, fst_c, None, None, 1)
	p5 = proc_line(line_img, letter_size*1.5, fst_c, 0.8, 1.1, 0)
	p6 = proc_line(line_img, letter_size*2, fst_c, 0.8, 1.1, 0)
	p7 = proc_line(line_img, letter_size*0.6, fst_c, 0.8, 1.1, 0)
	#p8 = proc_line(line_img, letter_size*0.7, fst_c, 0.8, 1.1, 0)
	p9 = proc_line(line_img, letter_size*0.9, fst_c, 0.8, 1.1, 0)

	char_list = []
	for cp in zip(p1, p2, p3, p5, p6, p7, p9):
		if len(char_list) > 0:
			char_list.append(Char(None, CHARTYPE.BLANK))
		char_list.extend(compare_pass(cp))
	return char_list

if __name__ == "__main__":
	main()
