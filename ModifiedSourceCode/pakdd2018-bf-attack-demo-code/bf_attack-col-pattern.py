# Bloom filter attack using a pattern mining on columns based approach
#
# Peter Christen, Anushka Vidanage, Thilina Ranbaduge, and Rainer Schnell
# Nov 2016 to Feb 2018
# 
# Initial ideas developed at the Isaac Newton Instutute for Mathematical
# Science, Cambridge (UK), during the Data Linkage and Anonymisation programme.
#
# Copyright 2018 Australian National University and others.
# All Rights reserved.
#
# -----------------------------------------------------------------------------
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------------
#
# Usage:
#   python bf_attack.py [q] [hash_type] [num_hash_funct] [bf_len] [bf_harden]
#                       [stop_iter_perc] [min_part_size]
#                       [encode_data_set_name] [encode_rec_id_col]
#                       [encode_col_sep] [encode_header_line_flag]
#                       [encode_attr_list]
#                       [plain_data_set_name] [plain_rec_id_col]
#                       [plain_col_sep] [plain_header_line_flag]
#                       [plain_attr_list]
#                       [max_num_many] [re_id_method]
# where:
# q                         is the length of q-grams to use
# hash_type                 is either DH (double-hashing) or RH
#                           (random hashing)
# num_hash_funct            is a positive number or 'opt' (to fill BF 50%)
# bf_len                    is the length of Bloom filters
# bf_harden                 is either None, 'balance' or 'fold' for different
#                           BF hardening techniques
# stop_iter_perc            The minimum percentage difference required between
#                           the two most frequent q-grams to continue the
#                           recursive Apriori pattern mining approach
# min_part_size             The minimum number of BFs in a 'partition' for the
#                           partition to be used with the Apriori algorithm
# encode_data_set_name      is the name of the CSV file to be encoded into BFs
# encode_rec_id_col         is the column in the CSV file containing record
#                           identifiers
# encode_col_sep            is the character to be used to separate fields in
#                           the encode input file
# encode_header_line_flag   is a flag, set to True if the file has a header
#                           line with attribute (field) names
# encode_attr_list          is the list of attributes to encode and use for
#                           the linkage
# 
# plain_data_set_name       is the name of the CSV file to use plain text
#                           values from
# plain_rec_id_col          is the column in the CSV file containing record
#                           identifiers
# plain_col_sep             is the character to be used to separate fields in
#                           the plain text input file
# plain_header_line_flag    is a flag, set to True if the file has a header
#                           line with attribute (field) names
# plain_attr_list           is the list of attributes to get values from to
#                           guess if they can be re-identified
#
# max_num_many              For the re-identification step, the maximum number
#                           of 1-to-many matches to consider
# re_id_method              The approach to be used for re-identification, with
#                           possible values: 'set_inter', 'apriori',
#                           'q_gram_tuple', 'bf_q_gram_tuple', 'bf_tuple',
#                           'all', 'none' (if set to 'none' then no
#                           re-identification will be attempted)

# Note that if the plain text data set is the same as the encode data set
# (with the same attributes) then the encode data set will also be used as the
# plain text data set.

# Example call:
# python bf_attack-col-pattern.py 2 DH 10 1000 none 1.0 8000 data/ncvoter-20140619-temporal-balanced-ratio-1to1-a.csv.gz 0 ,
# True [1,2,3,7]  data/ncvoter-20140619-temporal-balanced-ratio-1to1-b.csv.gz 0 , True [1,2,3,7] 10 all

# python bf_attack-col-pattern.py 2 DH 15 1000 none 1.0 8000 data/encoded.csv.gz 0 , True [1,2,3,7]  data/pt 0 , True [1,2,3,7] 10 all
# python2 bf_attack-col-pattern.py 2 RH 15 1000 fold 1.0 8000 data/encoded.csv 0 , False [0] data/plaintextQIDs.csv 0 , True [1,2,3,4,5] 10 all
# python2 bf_attack-col-pattern.py 2 RH 15 1000 fold 1.0 8000 data/plaintextQIDs.csv 0 , True [1,2,3,4,5] data/plaintextQIDs.csv 0 , True [1,2,3,4,5] 10 all

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

MAX_MEMORY_USE = 60000  # In Megabytes

# The difference between the average number of positions in the identified
# q-grams to be accepted, if less than that print a warning and don't consider
#
CHECK_POS_TUPLE_SIZE_DIFF_PERC = 20.0

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import csv
import gzip
import hashlib
import itertools
import math
#import os
import os.path
import random
import sys
import time

import auxiliary

import bitarray

import numpy
import io
import base64

DO_PADDING = False  # R. Schnell: Padding provides freq information to attack
PAD_CHAR = chr(1)   # Used for q-gram padding

BF_HASH_FUNCT1 = hashlib.md5
BF_HASH_FUNCT2 = hashlib.sha1

random.seed(42)

today_str = time.strftime("%Y%m%d", time.localtime())

NUM_SAMPLE = 1000  # Number of pairs / triplets to sample when calculating
                   # HW for bit positions

# -----------------------------------------------------------------------------
# Functions for step 1: Loading the data sets
# -----------------------------------------------------------------------------


def iter_decode(it):
    for line in it:
        yield line.decode('UTF-8')

def load_data_set_extract_q_grams(file_name, rec_id_col, use_attr_list,
                                  col_sep_char, header_line, q, padded):
  """Load the given file, extract selected attributes and convert them into one
     q-gram set per record.

     If more than one attribute are given we concatenate them with a space (so
     we only consider one single value per record).

     Returns:
     1) a dictionary with keys being record identifiers and values being their
        selected attribute values (if several concatenated as one string),
     2) a dictionary with keys being record identifiers and values being sets
        of q-grams,
     3) a set of all unique q-grams extracted,
     4) the average and
     5) standard deviation of the number of q-grams per record,
     6) a dictionary with the frequency of each q-gram (number of records it
        occurs in),
     7) a dictionary which for each q-gram has a set of all attribute values
        it occurs in (the concatenated values if several attributes are used),
     8) a dictionary which for each attribute value contains its set of
        record identifiers,
     9) and a list of the attribute names used.
  """

  start_time = time.time()

  if (file_name.endswith('gz')):
    f = gzip.open(file_name)
  else:
    f = io.open(file_name, 'r', encoding="UTF-8", newline='', errors='ignore')

  csv_reader = csv.reader(f, delimiter=col_sep_char, lineterminator='\n')

  rows = list(csv_reader)

  print 'Load data set from file:', file_name
  print '  Attribute separator: %c' % (col_sep_char)
  if (header_line == True):
    header_list = rows[0]
    print '  Header line:', header_list
  csv_reader = rows

  use_attr_name_list = []

  if (header_line == True):
    print '  Record identifier attribute:', header_list[rec_id_col]
  else:
    print '  Record identifier attribute number:', rec_id_col
  if (header_line == True):
    print '  Attributes to use:',
    for attr_num in use_attr_list:
      use_attr_name = header_list[attr_num]
      print use_attr_name,
      use_attr_name_list.append(use_attr_name)
  print
  print '  Extract q-grams with q=%d' % (q)
  print

  qm1 = q-1  # Shorthand

  rec_val_dict = {}  # For each record the extracted attribute value(s) as set

  q_gram_dict = {}  # One set of q-grams per input record (unless no q-grams
                    # in a record)

  q_gram_freq_dict = {}  # Number of records a q-gram occurs in

  q_gram_attr_val_dict = {}  # Attribute values for all q-grams

  all_unique_q_gram_set = set()

  num_q_gram_per_rec_list = []  # To calculate averages and std-deviations

  attr_val_freq_dict =   {}  # Number of records per attrbute value

  attr_val_rec_id_dict = {}  # Keep all record identifiers for each attribute
                             # value

  shortest_attr_val = 'x'*99
  longest_attr_val =   ''

  rec_num = 0

  # Attribute values from which no q-grams can be extracted
  #
  short_attr_val_set = set()
  for rec_list in csv_reader[1:]:
    rec_num += 1

    if (rec_num % 100000 == 0):
      time_used = time.time() - start_time
      print '  Processed %d records in %d sec (%.2f msec average)' % \
            (rec_num, time_used, 1000.0*time_used/rec_num)
      print '   ', auxiliary.get_memory_usage()

      auxiliary.check_memory_use(MAX_MEMORY_USE)

    rec_id = rec_list[rec_id_col].strip().lower()

    # Only keep record identifier as a number (remove 'org' 'dup-X' for
    # synthetic data sets)
    #
    if '-' in rec_id:
      rec_id = rec_id.split('-')[1].strip()

    if rec_id in q_gram_dict:
      continue  # If there are duplicate records for this identifier skip it

    attr_val_list = []  # All attributes to be used from this record

    for attr_num in use_attr_list:
      attr_val_list.append(rec_list[attr_num].strip().lower())

    attr_val = ' '.join(attr_val_list)  # Make it one string per record

    rec_val_dict[rec_id] = attr_val

    attr_val_len = len(attr_val)

    # Check if the attribute value is long enough to generate at least one
    # q-gram (with padding we get long enough attribute values)
    #
    if ((padded == True) or ((padded == False) and (attr_val_len >= q))):
      if attr_val_len < len(shortest_attr_val):
        shortest_attr_val = attr_val
      if attr_val_len > len(longest_attr_val):
        longest_attr_val = attr_val

      if (padded == True):  # Add padding start and end characters
        attr_val = PAD_CHAR*qm1+attr_val+PAD_CHAR*qm1
        attr_val_len += 2*qm1

      # Keep track of records that have this attribute value as well as the
      # frequency of attribute values
      #
      attr_val_rec_id_set = attr_val_rec_id_dict.get(attr_val, set())
      attr_val_rec_id_set.add(rec_id)
      attr_val_rec_id_dict[attr_val] = attr_val_rec_id_set

      attr_val_freq_dict[attr_val] = attr_val_freq_dict.get(attr_val, 0) + 1

      # Convert into q-grams and process them
      #
      attr_q_gram_list = [attr_val[i:i+q] for i in range(attr_val_len - qm1)]
      attr_q_gram_set = set(attr_q_gram_list)

      for q_gram in attr_q_gram_set:

        # For each q-gram collect all its attribute values
        #
        q_gram_attr_val_set = q_gram_attr_val_dict.get(q_gram, set())
        q_gram_attr_val_set.add(attr_val)
        q_gram_attr_val_dict[q_gram] = q_gram_attr_val_set

      q_gram_dict[rec_id] = attr_q_gram_set

      for q_gram in attr_q_gram_set:
        all_unique_q_gram_set.add(q_gram)
        q_gram_freq_dict[q_gram] = q_gram_freq_dict.get(q_gram, 0) + 1

      num_q_gram_per_rec_list.append(len(attr_q_gram_set))

    else:  # Attribute value too short to extract q-grams
      short_attr_val_set.add(attr_val)

  if (len(short_attr_val_set) > 0):
    print '  *** Warning, %d records contained attribute values to short' % \
          (len(short_attr_val_set)) + ' to generate q-grams ***'
    print '    ', sorted(short_attr_val_set)[:20], '...'
    print

  elif (len(q_gram_dict) < rec_num):
    print '  *** Warning, data set contains %d duplicates (same identifier)' \
          % (rec_num - len(q_gram_dict)) + ' or empty records ***'
    print'       %d unique records' % (len(q_gram_dict))
    print

  time_used = time.time() - start_time
  print '  Processed %d records in %d sec (%.2f msec average)' % \
        (rec_num, time_used, 1000.0*time_used/rec_num)
  print '   ', auxiliary.get_memory_usage()

  avrg_num_q_gram = numpy.mean(num_q_gram_per_rec_list)
  stdd_num_q_gram = numpy.std(num_q_gram_per_rec_list)

  print '  Found %d unique q-grams' % (len(all_unique_q_gram_set))
  print '    Average number of q-grams per record and std-dev: %d / %.2f' % \
        (avrg_num_q_gram, stdd_num_q_gram)
  print '      Minimum and maximum number of q-grams: %d / %d' % \
        (min(num_q_gram_per_rec_list), max(num_q_gram_per_rec_list))
  print '    Most frequent q-grams:'
  most_freq_q_gram_list = sorted(q_gram_freq_dict.items(),
                                 key=lambda t: t[1], reverse=True)
  print '     ', most_freq_q_gram_list[:5], '...'
  print '     ', most_freq_q_gram_list[5:10]
  print

  # Print frequent attribute values
  #
  attr_val_freq_list = sorted(attr_val_freq_dict.items(), key=lambda t: t[1], \
                              reverse=True)
  print '  Most frequent of %d unique attribute values:' % \
        (len(attr_val_freq_dict))
  print '   ', attr_val_freq_list[:5], '...'
  print '   ', attr_val_freq_list[5:10]
  print '  Shortest and longest attribute value:'
  print '    "%s"' % (shortest_attr_val)
  print '    "%s"' % (longest_attr_val)
  print

  attr_val_freq_dict = {}
  del num_q_gram_per_rec_list

  return rec_val_dict, q_gram_dict, all_unique_q_gram_set, \
         avrg_num_q_gram, stdd_num_q_gram, q_gram_freq_dict, \
         q_gram_attr_val_dict, attr_val_rec_id_dict, use_attr_name_list

# -----------------------------------------------------------------------------

def gen_q_gram_supp_graph(unique_q_gram_set, q_gram_dict, min_supp=None):
  """From the given set of all unique q-grams and dictionary with q-grams sets
     from records, generate a graph where nodes are q-grams and edges are
     connection q-grams if two q-grams occur in the same record.

     The attribute values of nodes and edges are their support as the number of
     records they occur in (either a single q-gram for nodes, or pairs of
     q-grams for edges).

     If a value is given for 'min_supp' (a float between 0 and 1) then only
     those nodes and edges that have this minimum value (with regard to the
     number of records in the given q-gram dictionary) are kept in the graph
     (dictionaries) returned.

     The function returns two dictionaries, one for nodes (q-grams as keys and
     support as values, and the other for edges (pairs of q-grams as keys and
     support as values).
  """

  if (min_supp != None):
    assert (0.0 < min_supp) and (1.0 >= min_supp), min_supp

  q_gram_node_dict = {}
  q_gram_edge_dict = {}

  for q_gram in unique_q_gram_set:
    q_gram_node_dict[q_gram] = 0

  for q_gram_set in q_gram_dict.itervalues():

    for q_gram in q_gram_set:  # Increase node support
      q_gram_node_dict[q_gram] += 1

    q_gram_list = sorted(q_gram_set)

    num_q_gram_pairs = 0

    for (i, q_gram1) in enumerate(q_gram_list[:-1]):

      for q_gram2 in q_gram_list[i+1:]:
        num_q_gram_pairs += 1

        q_gram_pair = (q_gram1, q_gram2)
        q_gram_edge_dict[q_gram_pair] = q_gram_edge_dict.get(q_gram_pair,0) + 1

    assert num_q_gram_pairs == len(q_gram_list)*(len(q_gram_list)-1)/2

  print 'Generated q-gram graph with %d nodes (q-grams) and %d edges' % \
        (len(q_gram_node_dict), len(q_gram_edge_dict))
  print '  Most common q-grams:'
  print '   ', sorted(q_gram_node_dict.items(), key=lambda t: t[1],
        reverse=True)[:5], '...'
  print '   ', sorted(q_gram_node_dict.items(), key=lambda t: t[1],
        reverse=True)[5:10]
  print '  Most common q-gram pairs (edges):'
  print '   ', sorted(q_gram_edge_dict.items(), key=lambda t: t[1],
        reverse=True)[:5], '...'
  print '   ', sorted(q_gram_edge_dict.items(), key=lambda t: t[1],
        reverse=True)[5:10]
  print

  if (min_supp != None):  # Keep only those with minimum support
    num_rec = float(len(q_gram_dict))

    for (q_gram, supp) in q_gram_node_dict.items():
      if (float(supp) / num_rec < min_supp):
        del q_gram_node_dict[q_gram]

    for (q_gram_pair, supp) in q_gram_edge_dict.items():
      if (float(supp) / num_rec < min_supp):
        del q_gram_edge_dict[q_gram_pair]

    print '  After filtering with minimum support of %.1f%% (%d)' % \
          (100.0*min_supp ,int(min_supp*num_rec)),
    print 'the q-gram graph contains %d nodes (q-grams) and %d edges' % \
          (len(q_gram_node_dict), len(q_gram_edge_dict))
    print '    Most common q-grams:'
    print '     ', sorted(q_gram_node_dict.items(), key=lambda t: t[1],
          reverse=True)[:5], '...'
    print '     ', sorted(q_gram_node_dict.items(), key=lambda t: t[1],
          reverse=True)[5:10]
    print '    Most common q-gram pairs (edges):'
    print '     ', sorted(q_gram_edge_dict.items(), key=lambda t: t[1],
          reverse=True)[:5], '...'
    print '     ', sorted(q_gram_edge_dict.items(), key=lambda t: t[1],
          reverse=True)[5:10]
    print

  return q_gram_node_dict, q_gram_edge_dict

# -----------------------------------------------------------------------------
# Functions for step 2: Generating Bloom filters
# -----------------------------------------------------------------------------

def gen_bloom_filter_dict(q_gram_dict, hash_type, bf_len, num_hash_funct):
  """Encode the q-gram sets for all records in the given q-gram set dictionary
     into Bloom filters of the given length using the given number of hash
     functions.

     Return a dictionary with bit-patterns each of length of the given Bloom
     filter length, as well as a dictionary which for each bit postion contains
     the q-grams mapped to this position (the true mapping data).
  """

  print 'Generate Bloom filter bit-patterns for %d q-gram sets' % \
        (len(q_gram_dict))
  print '  Bloom filter length:          ', bf_len
  print '  Number of hash functions used:', num_hash_funct
  print '  Hashing type used:            ', \
        {'dh':'Double hashing', 'rh':'Random hashing'}[hash_type]

  bf_dict= {}  # One BF per record

  bf_pos_map_dict = {}  # For each bit position the q-grams mapped to it

  bf_num_1_bit_list = [] # Keep number of bits set to calculate avrg and std

  bf_pos_num_1_bit_list = [0]*bf_len  # Number of 1-bits per BF bit position

  start_time = time.time()

  rec_num = 0

  bf_len_m1 = bf_len-1
  file_name = 'data/plaintextQIDs.csv'

  f = io.open(file_name, 'r', encoding="UTF-8", newline='', errors='ignore')

  csv_reader = csv.reader(f, delimiter=',', lineterminator='\n')
  next(csv_reader, None)
  rows = list(csv_reader)
  csv_dict = {}
  for row in rows:
    csv_dict.update(({row[0]: row[6]}))
  for (rec_id, rec_q_gram_set) in q_gram_dict.iteritems():
    rec_num += 1

    if (rec_num % 100000 == 0):
      time_used = time.time() - start_time
      print '  Generated %d Bloom filters in %d sec (%.2f msec average)' % \
            (rec_num, time_used, 1000.0*time_used/rec_num)
      print '   ', auxiliary.get_memory_usage()

      auxiliary.check_memory_use(MAX_MEMORY_USE)

    rec_bf = bitarray.bitarray(bf_len)
    rec_bf.setall(0)

    # Hash all q-grams into bits in the Bloom filter
    #
    for q_gram in rec_q_gram_set:

      if (hash_type == 'dh'):  # Double hashing
        hex_str1 = BF_HASH_FUNCT1(q_gram).hexdigest()
        int1 =     int(hex_str1, 16)

        hex_str2 = BF_HASH_FUNCT2(q_gram).hexdigest()
        int2 =     int(hex_str2, 16)

        for i in range(num_hash_funct):
          gi = int1 + i*int2
          gi = int(gi % bf_len)

          if (rec_bf[gi] == 0):  # Not yet set
            rec_bf[gi] = 1
            bf_pos_num_1_bit_list[gi] += 1

          bf_pos_q_gram_set = bf_pos_map_dict.get(gi, set())
          bf_pos_q_gram_set.add(q_gram)
          bf_pos_map_dict[gi] = bf_pos_q_gram_set

      elif (hash_type == 'rh'):  # Random hashing
        hex_str = BF_HASH_FUNCT1(q_gram).hexdigest()
        random_seed = random.seed(int(hex_str,16))

        for i in range(num_hash_funct):
          gi = random.randint(0, bf_len_m1)

          if (rec_bf[gi] == 0):  # Not yet set
            rec_bf[gi] = 1
            bf_pos_num_1_bit_list[gi] += 1

          bf_pos_q_gram_set = bf_pos_map_dict.get(gi, set())
          bf_pos_q_gram_set.add(q_gram)
          bf_pos_map_dict[gi] = bf_pos_q_gram_set

      else:  # Should not happend
        raise Exception, hash_type

    current_bf = bitarray.bitarray()
    current_bf.setall(0)
    current_bf64 = csv_dict[rec_id]
    #a.endian = 'big'
    current_bf.frombytes(base64.b64decode(current_bf64))
    bf_dict[rec_id] = current_bf
    bf_num_1_bit_list.append(int(current_bf.count(1)))

  q_gram_map_list = []
  for bf_pos_q_gram_set in bf_pos_map_dict.values():
    q_gram_map_list.append(len(bf_pos_q_gram_set))

  print '  Bloom filter generation took %d sec' % (time.time()-start_time)
  print '    Number of 1-bits per BF (min / avr, std / max): ' + \
        '%d / %.2f, %.2f / %d' % (min(bf_num_1_bit_list),
                                  numpy.mean(bf_num_1_bit_list),
                                  numpy.std(bf_num_1_bit_list),
                                  max(bf_num_1_bit_list))
  print '    Number of 1-bits per BF position (min / avr, std / max): ' + \
        '%d / %.2f, %.2f / %d' % (min(bf_pos_num_1_bit_list),
                                  numpy.mean(bf_pos_num_1_bit_list),
                                  numpy.std(bf_pos_num_1_bit_list),
                                  max(bf_pos_num_1_bit_list))
  print '    Minimum, average and maximum number of q-grams mapped to a ' + \
        'bit position: %d / %.2f / %d' % (min(q_gram_map_list),
        numpy.mean(q_gram_map_list), max(q_gram_map_list))
  print '   ', auxiliary.get_memory_usage()
  print

  del bf_num_1_bit_list
  del q_gram_map_list

  assert len(bf_dict) == len(q_gram_dict)

  return bf_dict, bf_pos_map_dict 

# -----------------------------------------------------------------------------

def balance_bf_dict(bf_dict, bf_len, true_q_gram_pos_map_dict):
  """Balance all Bloom filters in the given dictionary by concatenating each BF
     with its complement (negated BF), doubling the length of all BFs.

     The true position dictionary will also be modified.
  """

  print 'Balance Bloom filters of length %d' % (bf_len)
  print

  for rec_id in bf_dict:
    this_bf = bf_dict[rec_id]

    bf_dict[rec_id] = this_bf + ~this_bf

    assert bf_dict[rec_id].count(1) == bf_dict[rec_id].count(0)

    assert len(bf_dict[rec_id]) == 2*bf_len, (len(bf_dict[rec_id]), 2*bf_len)

#  print 0, sorted(true_q_gram_pos_map_dict[0])
#  print 1, sorted(true_q_gram_pos_map_dict[1])
#  print 99, sorted(true_q_gram_pos_map_dict[99])

  # Adjust the position dictionary (for all positions x generate a position
  # x + bf_len and assign the same q-gram set to it)
  #
  for (pos, q_gram_set) in true_q_gram_pos_map_dict.items():
    new_pos = pos+bf_len
    true_q_gram_pos_map_dict[new_pos] = q_gram_set.copy()

  assert sorted(true_q_gram_pos_map_dict[0]) == \
         sorted(true_q_gram_pos_map_dict[0+bf_len])
  assert sorted(true_q_gram_pos_map_dict[1]) == \
         sorted(true_q_gram_pos_map_dict[1+bf_len])
  assert sorted(true_q_gram_pos_map_dict[99]) == \
         sorted(true_q_gram_pos_map_dict[99+bf_len])
  assert sorted(true_q_gram_pos_map_dict[bf_len-1]) == \
         sorted(true_q_gram_pos_map_dict[bf_len-1+bf_len])

#  print 0, sorted(true_q_gram_pos_map_dict[0])
#  print 1, sorted(true_q_gram_pos_map_dict[1])
#  print 99, sorted(true_q_gram_pos_map_dict[99])
#  print 0+bf_len, sorted(true_q_gram_pos_map_dict[0+bf_len])
#  print 1+bf_len, sorted(true_q_gram_pos_map_dict[1+bf_len])
#  print 99+bf_len, sorted(true_q_gram_pos_map_dict[99+bf_len])

  return bf_dict, true_q_gram_pos_map_dict

# -----------------------------------------------------------------------------

def fold_bf_dict(bf_dict, bf_len, true_q_gram_pos_map_dict):
  """Split all Bloom filters in half then XOR the two halfs, resulting in new
     BF half as long as the original ones.

     The true position dictionary will also be modified.
  """

  print 'Fold and XOR Bloom filters'
  print

  half_len = int(bf_len / 2)

  for rec_id in bf_dict:
    this_bf = bf_dict[rec_id]

    bf_dict[rec_id] = this_bf[:half_len] ^ this_bf[half_len:]

#  print 0, sorted(true_q_gram_pos_map_dict[0])
#  print 1, sorted(true_q_gram_pos_map_dict[1])
#  print 99, sorted(true_q_gram_pos_map_dict[99])
#  print 0+half_len, sorted(true_q_gram_pos_map_dict[0+half_len])
#  print 1+half_len, sorted(true_q_gram_pos_map_dict[1+half_len])
#  print 99+half_len, sorted(true_q_gram_pos_map_dict[99+half_len])

  # Adjust the position dictionary (all positions x larger than bf_len/2 are
  # mapped to x - bf_len/2)
  #
  for (pos, q_gram_set) in true_q_gram_pos_map_dict.items():
    if (pos >= half_len):
      org_pos_q_gram_set = true_q_gram_pos_map_dict[pos]
      new_pos = pos - half_len
      new_pos_q_gram_set = true_q_gram_pos_map_dict[new_pos]
      new_pos_q_gram_set = new_pos_q_gram_set.union(org_pos_q_gram_set)
      true_q_gram_pos_map_dict[new_pos] = new_pos_q_gram_set
      del true_q_gram_pos_map_dict[pos]

#  assert sorted(true_q_gram_pos_map_dict[0]) == \
#         sorted(true_q_gram_pos_map_dict[0].union(true_q_gram_pos_map_dict[0+half_len]))
#  assert sorted(true_q_gram_pos_map_dict[1]) == \
#         sorted(true_q_gram_pos_map_dict[1].union(true_q_gram_pos_map_dict[1+half_len]))
#  assert sorted(true_q_gram_pos_map_dict[99]) == \
#         sorted(true_q_gram_pos_map_dict[99].union(true_q_gram_pos_map_dict[99+half_len]))

#  print 0, sorted(true_q_gram_pos_map_dict[0])
#  print 1, sorted(true_q_gram_pos_map_dict[1])
#  print 99, sorted(true_q_gram_pos_map_dict[99])

  return bf_dict, true_q_gram_pos_map_dict

# -----------------------------------------------------------------------------

def gen_bf_col_dict(bf_dict, bf_len):
  """Convert the given BF dictionary into a column-wise format as a list,
     where each column will be a bit array.

     Returns this list of bit arrays as well as a list of how the original
     records refer to elements in bit positions lists (i.e. which entry in a
     bit position list corresponds to which encoded BF.
  """

  num_bf = len(bf_dict)

  bit_col_list = []  # One bit array per position

  start_time = time.time()

  for bit_pos in range(bf_len):
    bit_col_list.append(bitarray.bitarray(num_bf))

  rec_id_list = sorted(bf_dict.keys())

  # Fill newly created bit position arrays
  #
  for (rec_num, rec_id) in enumerate(rec_id_list):
    rec_bf = bf_dict[rec_id]

    for pos in range(bf_len):
      bit_col_list[pos][rec_num] = rec_bf[pos]

  # Check both BF dict and column-wise BF arrays are the same
  #
  rec_id_bf_list = sorted(bf_dict.items())  # One BF per record

  for pos in range(bf_len):
    for rec_num in range(num_bf):
      assert rec_id_bf_list[rec_num][1][pos] == bit_col_list[pos][rec_num]

  bf_pos_num_1_bit_list = []
  for bit_array in bit_col_list:
    bf_pos_num_1_bit_list.append(int(bit_array.count(1)))

  print 'Generated column-wise BF storage'
  print '  Number of 1-bits per BF position (min / avr, std / max): ' + \
        '%d / %.2f, %.2f / %d' % (min(bf_pos_num_1_bit_list),
                                  numpy.mean(bf_pos_num_1_bit_list),
                                  numpy.std(bf_pos_num_1_bit_list),
                                  max(bf_pos_num_1_bit_list))
  print '  As percentages of all BFs: %.2f%% / %.2f%%, %.2f%% / %.2f%%' % \
        (100.0*min(bf_pos_num_1_bit_list)/num_bf,
         100.0*numpy.mean(bf_pos_num_1_bit_list)/num_bf,
         100.0*numpy.std(bf_pos_num_1_bit_list)/num_bf,
         100.0*max(bf_pos_num_1_bit_list)/num_bf)

  print '  Time to generate column-wise BF bit arrays: %.2f sec' % \
        (time.time() - start_time)
  print '   ', auxiliary.get_memory_usage()
  print

  return bit_col_list, rec_id_list

# -----------------------------------------------------------------------------

def get_bf_row_col_freq_dist(bf_dict, bit_col_list):
  """Calculate how often each unique BF row and column pattern occurs.

     Return two dictionaries with row and column frequencies of counts of
     occurrences.
  """

  num_bf = len(bf_dict)
  bf_len = len(bit_col_list)

  row_freq_dict = {}
  col_freq_dict = {}

  for bf in bf_dict.itervalues():
    bf_str = bf.to01()
    row_freq_dict[bf_str] = row_freq_dict.get(bf_str, 0) + 1

  for bf_bit_pos in bit_col_list:
    bf_str = bf_bit_pos.to01()
    col_freq_dict[bf_str] = col_freq_dict.get(bf_str, 0) + 1

  row_count_dict = {}  # Now count how often each frequency occurs
  col_count_dict = {}

  for freq in row_freq_dict.itervalues():
    row_count_dict[freq] = row_count_dict.get(freq, 0) + 1
  for freq in col_freq_dict.itervalues():
    col_count_dict[freq] = col_count_dict.get(freq, 0) + 1

  print 'BF frequency distribution:'
  for (freq, count) in sorted(row_count_dict.items(),
                              key=lambda t: (t[1],t[0]),
                              reverse=True):
    if (count == 1):
      print '        1 BF pattern occurs %d times' % (freq)
    else:
      print '  %6d BF patterns occur %d times' % (count, freq)

  print 'BF bit position frequency distribution:'
  for (freq, count) in sorted(col_count_dict.items(),
                              key=lambda t: (t[1],t[0]),
                              reverse=True):
    if (count == 1):
      print '        1 BF bit position pattern occurs %d times' % (freq)
    else:
      print '  %6d BF bit position patterns occur %d times' % (count, freq)
  print

  return row_count_dict, col_count_dict

# -----------------------------------------------------------------------------

def check_balanced_bf(bf_bit_pos_list):
  """Check if the given column-wise list of bit arrays was generated using a
     BF balancing hardening approach - i.e. if pairs of bit positions have the
     same pattern if one bit array is negated.

     Return a list with pairs of bit positions that are the same if one is
     negated.
  """

  bit_pos_pair_set = set()

  num_bf = len(bf_bit_pos_list[0])
  bf_len = len(bf_bit_pos_list)

  # Loop over all pairs of bit positions (columns)
  #
  for (pos1, bit_array_pos1) in enumerate(bf_bit_pos_list[:-1]):
    for pos2 in range(pos1+1,bf_len):
      bit_array_pos2 = bf_bit_pos_list[pos2]

      # If the second bit column is the complement of the first then XOR
      # should give only 1-bits
      #
      xor_bit_pair_array = bit_array_pos1 ^ bit_array_pos2

      if (int(xor_bit_pair_array.count(1)) == num_bf):
        bit_pos_pair_set.add((pos1,pos2))

  # Check how many pairs of bit positions were found
  #
  if (len(bit_pos_pair_set) == int(bf_len/2)):
    print 'Found %d bit position pairs that match -> BF set is balanced' % \
          (len(bit_pos_pair_set))
  else:
    print 'Found %d bit position pairs that match -> BF set is not balanced' \
          % (len(bit_pos_pair_set))+' (would need %d pairs)' % (int(bf_len/2))
  print

  # If the BFs were balanced, try to find the original columns as those that
  # have less 1-bits - this only works if the original columns contained less
  # than 50% 1-bits (i.e. less than half of records had a 1-bit in a position)
  #
  if (len(bit_pos_pair_set) == int(bf_len/2)):

    org_col_set = set()
    neg_col_set = set()

    for (pos1,pos2) in bit_pos_pair_set:
      hw1 = int(bf_bit_pos_list[pos1].count(1))
      hw2 = int(bf_bit_pos_list[pos2].count(1))

      if (hw1 < hw2):
        org_col_set.add(pos1)
        neg_col_set.add(pos2)
      else:
        org_col_set.add(pos2)
        neg_col_set.add(pos1)

    # ***********************************************************************
    # TODO PC 20170920: Does not work, the above assumption does not hold
    # ***********************************************************************

    print '  Original bit positions seem to be:', sorted(org_col_set)
    print '  Negated bit positions seem to be: ', sorted(neg_col_set)

  return bit_pos_pair_set

# -----------------------------------------------------------------------------

def check_hamming_weight_bit_positions(bf_bit_pos_list, num_sample):
  """For the given list of bit position bit arrays (column-wise BFs), calculate
     and print the average Hamming weight (HW) for pairs and triplets of
     randomly selected positions using both AND and XOR operations between bit
     arrays.
  """

  bit_pos_pair_and_dict = {}  # Keys are pairs of bit positions
  bit_pos_pair_xor_dict = {}

  bit_pos_triplet_and_dict = {}
  bit_pos_triplet_xor_dict = {}

  bf_len = len(bf_bit_pos_list)
  num_rec= len(bf_bit_pos_list[0])

  bit_pos_list = range(bf_len) # Position numbers to sample from

  while (len(bit_pos_pair_and_dict) < num_sample):
    bit_pos_pair = tuple(random.sample(bit_pos_list, 2))

    if (bit_pos_pair not in bit_pos_pair_and_dict):  # A new position pair
      pos1, pos2 = bit_pos_pair
      and_bit_array = bf_bit_pos_list[pos1] & bf_bit_pos_list[pos2]  # AND
      xor_bit_array = bf_bit_pos_list[pos1] ^ bf_bit_pos_list[pos2]  # XOR

      bit_pos_pair_and_dict[bit_pos_pair] = int(and_bit_array.count(1))
      bit_pos_pair_xor_dict[bit_pos_pair] = int(xor_bit_array.count(1))

  bit_pos_pair_and_hw_list = bit_pos_pair_and_dict.values()
  bit_pos_pair_xor_hw_list = bit_pos_pair_xor_dict.values()

  and_hw_mean = numpy.mean(bit_pos_pair_and_hw_list)
  and_hw_std =  numpy.std(bit_pos_pair_and_hw_list)
  xor_hw_mean = numpy.mean(bit_pos_pair_xor_hw_list)
  xor_hw_std =  numpy.std(bit_pos_pair_xor_hw_list)

  print 'Hamming weights between random pairs from %d samples and %d ' % \
        (num_sample, num_rec) + 'records:'
  print '  AND: %d (%.2f%%) average, %d std-dev (%.2f%%)' % \
        (and_hw_mean, 100.0*and_hw_mean/num_rec,
         and_hw_std,  100.0*and_hw_std/num_rec)
  print '  XOR: %d (%.2f%%) average, %d std-dev (%.2f%%)' % \
        (xor_hw_mean, 100.0*xor_hw_mean/num_rec,
         xor_hw_std,  100.0*xor_hw_std/num_rec)

  while (len(bit_pos_triplet_and_dict) < num_sample):
    bit_pos_triplet = tuple(random.sample(bit_pos_list, 3))

    if (bit_pos_triplet not in bit_pos_triplet_and_dict):  # A new triplet
      pos1, pos2, pos3 = bit_pos_triplet
      and_bit_array = bf_bit_pos_list[pos1] & bf_bit_pos_list[pos2]  # AND
      and_bit_array = and_bit_array & bf_bit_pos_list[pos3]
      xor_bit_array = bf_bit_pos_list[pos1] ^ bf_bit_pos_list[pos2]  # XOR
      xor_bit_array = xor_bit_array ^ bf_bit_pos_list[pos3]

      bit_pos_triplet_and_dict[bit_pos_triplet] = \
                                        int(and_bit_array.count(1))
      bit_pos_triplet_xor_dict[bit_pos_triplet] = \
                                        int(xor_bit_array.count(1))

  bit_pos_triplet_and_hw_list = bit_pos_triplet_and_dict.values()
  bit_pos_triplet_xor_hw_list = bit_pos_triplet_xor_dict.values()

  and_hw_mean = numpy.mean(bit_pos_triplet_and_hw_list)
  and_hw_std =  numpy.std(bit_pos_triplet_and_hw_list)
  xor_hw_mean = numpy.mean(bit_pos_triplet_xor_hw_list)
  xor_hw_std =  numpy.std(bit_pos_triplet_xor_hw_list)

  print 'Hamming weights between random triplets from %d samples and %d ' % \
        (num_sample, num_rec) + 'records:'
  print '  AND: %d (%.2f%%) average, %d std-dev (%.2f%%)' % \
        (and_hw_mean, 100.0*and_hw_mean/num_rec,
         and_hw_std,  100.0*and_hw_std/num_rec)
  print '  XOR: %d (%.2f%%) average, %d std-dev (%.2f%%)' % \
        (xor_hw_mean, 100.0*xor_hw_mean/num_rec,
         xor_hw_std,  100.0*xor_hw_std/num_rec)
  print
 
# -----------------------------------------------------------------------------
# Functions for step 3: Pattern mining to get frequent BF bit positions
# -----------------------------------------------------------------------------

def get_most_freq_other_q_grams(q_gram_dict, must_be_in_rec_q_gram_set,
                                must_not_be_in_rec_q_gram_set):
  """From the given q-gram dictionary and filter q-gram sets, get the frequent
     other q-grams (not in the filter sets), where each q-gram in the
     'must_be_in_rec_q_gram_set' must be in a record q-gram set for the record
     to be counted, and no q-gram in the 'must_not_be_in_rec_q_gram_set' must
     be in a record q-gram set for the record to be counted.

     Returns a list of tuples (q-gram, count) sorted according to their counts
     (most frequent first).
  """

  num_rec = len(q_gram_dict)

  num_rec_part = 0  # Number of records in this partition that are considered

  other_q_gram_freq_dict = {}

  for rec_q_gram_set in q_gram_dict.itervalues():

    # Check if the record q-gram set fulfills the in/out conditions

    # All q-grams in 'must_be_in_rec_q_gram_set' must occur in a record
    #
    all_must_in = must_be_in_rec_q_gram_set.issubset(rec_q_gram_set)

    # No q-gram in 'must_not_be_in_rec_q_gram_set' must occur in record
    #
    if (len(must_not_be_in_rec_q_gram_set.intersection(rec_q_gram_set)) == 0):
      all_must_not_out = True
    else:  # Non-empty intersection, so some q-grams are in both sets
      all_must_not_out = False

    if (all_must_in == True) and (all_must_not_out == True):
      num_rec_part += 1  # Consider this record

      for q_gram in rec_q_gram_set:
        if (q_gram not in must_be_in_rec_q_gram_set):
#        if ((q_gram not in must_be_in_rec_q_gram_set) and \
#            (q_gram not in must_not_be_in_rec_q_gram_set)):
          other_q_gram_freq_dict[q_gram] = \
                                      other_q_gram_freq_dict.get(q_gram,0) + 1

  # Get most frequent other q-grams
  #
  freq_q_gram_count_list = sorted(other_q_gram_freq_dict.items(),
                                  key=lambda t: t[1], reverse=True)

  print 'Most frequent other q-grams (from records containing %s and not' % \
        (str(must_be_in_rec_q_gram_set)) + ' containing %s):' % \
        (str(must_not_be_in_rec_q_gram_set))

  # Print 10 most frequent other q-grams
  #
  for (q_gram, count) in freq_q_gram_count_list[:10]:
    print '  %s: %d (%.2f%%, %.2f%%)' % (q_gram, count, 100.0*count/num_rec,
             100.0*count/num_rec_part)
  print

  return freq_q_gram_count_list

# -----------------------------------------------------------------------------

def gen_freq_bf_bit_positions(encode_bf_bit_pos_list, min_count,
                              col_filter_set=set(),
                              row_filter_bit_array=None,
                              verbose=False):
  """Using an Apriori based approach, find all individual, pairs, triplets,
     etc. of bit positions that occur frequently together in the given list of
     bit position arrays (column-wise BFs).

     Only consider bit positions (and pairs and tuples of them) that have a
     Hamming weight of at least 'min_count'.

     If 'col_filter_set' is given (not an empty set), then do not consider
     columns listed in the set.

     If 'row_filter_bit_array' is given (not None), then do not consider the
     rows (BFs) that have a 0-bit.

     Return a dictionary where keys are the longest found tuples made of bit
     positions (integers) and values their counts of occurrences.
  """

  num_bf = len(encode_bf_bit_pos_list[0])

  # If needed generate the row filter bit array - set all rows (BFs) in the
  # filter set to 1 so all are considered
  #
  if (row_filter_bit_array == None):
    row_filter_bit_array = bitarray.bitarray(num_bf)
    row_filter_bit_array.setall(1)

  if (row_filter_bit_array != None):
    part_size = int(row_filter_bit_array.count(1))
  else:
    part_size = num_bf

  start_time = time.time()

  print 'Generate frequent bit position sets with HW of at least %d' % \
        (min_count)
  print '  Partiton size: %d Bfs (from %d total BFs)' % (part_size, num_bf)

  # The dictionary with frequent bit position tuples to be returned
  #
  freq_bf_bit_pos_dict = {}

  # First get all bit positions with a HW of at least 'min_count' - - - - - - -
  #
  freq_bit_pos_dict = {}

  ind_start_time = time.time()  # Time to get frequent individual positions

  max_count = -1

  for (pos, bit_array) in enumerate(encode_bf_bit_pos_list):

    # Only consider columns not given in the column filter set
    #
    if (pos not in col_filter_set):

      # Filter (AND) with row filter bit array
      #
      bit_pos_array_filtered = bit_array & row_filter_bit_array

      bit_pos_hw = int(bit_pos_array_filtered.count(1))
      max_count = max(max_count, bit_pos_hw)
      if (bit_pos_hw >= min_count):
        freq_bit_pos_dict[pos] = bit_pos_hw
        freq_bf_bit_pos_dict[(pos,)] = bit_pos_hw

  print '  Found %d bit positions with a HW of at least %d (from %d BFs):' % \
        (len(freq_bit_pos_dict), min_count, num_bf)

  if (verbose == True):  # Print the 10 most and least frequent ones
    if (len(freq_bit_pos_dict) <= 20):
      for (pos, count) in sorted(freq_bit_pos_dict.items(),
                                 key=lambda t: t[1], reverse=True):
        print '    %d: %d (%.2f%% / %.2f%%)' % \
              (pos, count, 100.0*count/num_bf, 100.0*count/part_size)
    else:
      for (pos, count) in sorted(freq_bit_pos_dict.items(),
                                 key=lambda t: t[1], reverse=True)[:10]:
        print '    %d: %d (%.2f%% / %.2f%%)' % \
              (pos, count, 100.0*count/num_bf, 100.0*count/part_size)
      print '        ....'
      for (pos, count) in sorted(freq_bit_pos_dict.items(),
                                 key=lambda t: t[1], reverse=True)[-10:]:
        print '    %d: %d (%.2f%% / %.2f%%)' % \
              (pos, count, 100.0*count/num_bf, 100.0*count/part_size)
    print

  print '  Generation of frequent individual BF bit positions took %.2f sec' \
        % (time.time()-ind_start_time)
  print '   ', auxiliary.get_memory_usage()
  print

  auxiliary.check_memory_use(MAX_MEMORY_USE)

  # Next get all pairs of bit positions with a HW of at least 'min_count' - -
  #
  freq_bit_pos_pair_dict = {}

  pair_start_time = time.time()  # Time to get frequent pairs of positions

  freq_bit_pos_list = sorted(freq_bit_pos_dict.keys())

  for (i, pos1) in enumerate(freq_bit_pos_list[:-1]):
    for pos2 in freq_bit_pos_list[i+1:]:
      assert pos1 < pos2, (pos1, pos2)

      # Filter (AND) with row filter bit array
      #
      bit_array_pos1_filt = encode_bf_bit_pos_list[pos1] & row_filter_bit_array

      bit_array_pos2 =     encode_bf_bit_pos_list[pos2]
      and_bit_pair_array = bit_array_pos1_filt & bit_array_pos2

      and_bit_pos_pair_hw = int(and_bit_pair_array.count(1))

      if (and_bit_pos_pair_hw >= min_count):
        freq_bit_pos_pair_dict[(pos1,pos2)] = and_bit_pos_pair_hw

  if (len(freq_bit_pos_pair_dict) == 0):  # No frequent pairs, return frequent
    return freq_bf_bit_pos_dict           # individuals

  freq_bf_bit_pos_dict = freq_bit_pos_pair_dict  # To be returned

  print '  Found %d bit position pairs with a HW of at least ' % \
        (len(freq_bit_pos_pair_dict)) + '%d (from %d BFs):' % \
        (min_count, num_bf)

  if (verbose == True):  # Print the 10 most and least frequent pairs
    if (len(freq_bit_pos_pair_dict) <= 20):
      for (pos_pair, count) in sorted(freq_bit_pos_pair_dict.items(),
                                 key=lambda t: t[1], reverse=True):
        print '    %s: %d (%.2f%% / %.2f%%)' % \
              (str(pos_pair), count, 100.0*count/num_bf, 100.0*count/part_size)
    else:
      for (pos_pair, count) in sorted(freq_bit_pos_pair_dict.items(),
                                 key=lambda t: t[1], reverse=True)[:10]:
        print '    %s: %d (%.2f%% / %.2f%%)' % \
              (str(pos_pair), count, 100.0*count/num_bf, 100.0*count/part_size)
      print '        ....'
      for (pos_pair, count) in sorted(freq_bit_pos_pair_dict.items(),
                                      key=lambda t: t[1], reverse=True)[-10:]:
        print '    %s: %d (%.2f%% / %.2f%%)' % \
              (str(pos_pair), count, 100.0*count/num_bf, 100.0*count/part_size)
    print

  print '  Generation of frequent pairs of BF bit positions took %.2f sec' % \
        (time.time()-pair_start_time)
  print '   ', auxiliary.get_memory_usage()
  print

  auxiliary.check_memory_use(MAX_MEMORY_USE)

  prev_freq_bit_pos_tuple_dict = freq_bit_pos_pair_dict

  # Now run Apriori for sets of size 3 and more
  #
  curr_len_m1 = 1
  curr_len_p1 = 3

  while (len(prev_freq_bit_pos_tuple_dict) > 1):
    prev_freq_bit_pos_tuple_list = sorted(prev_freq_bit_pos_tuple_dict.keys())

    loop_start_time = time.time()

    # Generate candidates of current length plus 1
    #
    cand_bit_pos_tuple_dict = {}

    for (i, pos_tuple1) in enumerate(prev_freq_bit_pos_tuple_list[:-1]):
      pos_tuple1_m1 =   pos_tuple1[:curr_len_m1]
      pos_tuple1_last = pos_tuple1[-1]

      for pos_tuple2 in prev_freq_bit_pos_tuple_list[i+1:]:

        # Check if the two tuples have the same beginning
        #
        if (pos_tuple1_m1 == pos_tuple2[:curr_len_m1]):
          assert pos_tuple1_last < pos_tuple2[-1], (pos_tuple1, pos_tuple2)
          cand_pos_tuple = pos_tuple1 + (pos_tuple2[-1],)

          # Check all sub-tuples are in previous frequent tuple set
          #
          all_sub_tuple_freq = True
          for pos in range(curr_len_p1):
            check_tuple = tuple(cand_pos_tuple[:pos] + \
                                cand_pos_tuple[pos+1:])
            if (check_tuple not in prev_freq_bit_pos_tuple_dict):
              all_sub_tuple_freq = False
              break

          if (all_sub_tuple_freq == True):  # Get intersection of positions
            and_bit_tuple_array = row_filter_bit_array
            for pos in cand_pos_tuple:
              and_bit_tuple_array = and_bit_tuple_array & \
                                                 encode_bf_bit_pos_list[pos]

            and_bit_pos_tuple_hw = int(and_bit_tuple_array.count(1))

            if (and_bit_pos_tuple_hw >= min_count):
              cand_bit_pos_tuple_dict[cand_pos_tuple] = and_bit_pos_tuple_hw

    if (len(cand_bit_pos_tuple_dict) == 0):
      break  # No more candidates, end Apriori process

    print '  Found %d bit position tuples of length %d with a HW of at ' % \
          (len(cand_bit_pos_tuple_dict), curr_len_p1) + \
          'least %d (from %d BFs):' % (min_count, num_bf)

    if (verbose == True):  # Print the 10 most and least frequent tuples
      if (len(cand_bit_pos_tuple_dict) <= 20):
        for (pos_tuple, count) in sorted(cand_bit_pos_tuple_dict.items(),
                                         key=lambda t: t[1], reverse=True):
          print '    %s: %d (%.2f%% / %.2f%%)' % \
                (str(pos_tuple),count, 100.0*count/num_bf,
                 100.0*count/part_size)
      else:
        for (pos_tuple, count) in sorted(cand_bit_pos_tuple_dict.items(),
                                         key=lambda t: t[1],
                                         reverse=True)[:10]:
          print '    %s: %d (%.2f%% / %.2f%%)' % \
                (str(pos_tuple),count, 100.0*count/num_bf,
                 100.0*count/part_size)
        print '        ....'
        for (pos_tuple, count) in sorted(cand_bit_pos_tuple_dict.items(),
                                         key=lambda t: t[1],
                                         reverse=True)[-10:]:
          print '    %s: %d (%.2f%% / %.2f%%)' % \
                (str(pos_tuple),count, 100.0*count/num_bf,
                 100.0*count/part_size)
      print

    print '  Generation of frequent BF bit position tuples took %.2f sec' % \
          (time.time()-loop_start_time)
    print '   ', auxiliary.get_memory_usage()
    print

    auxiliary.check_memory_use(MAX_MEMORY_USE)

    # Set found frequent bit position tuples as final dictionary
    #
    freq_bf_bit_pos_dict = cand_bit_pos_tuple_dict

    curr_len_m1 += 1
    curr_len_p1 += 1

    prev_freq_bit_pos_tuple_dict = cand_bit_pos_tuple_dict

  print 'Overall generation of frequent BF bit position sets took %.1f sec' % \
        (time.time()-start_time)
  print '  Identified %d frequent bit position sets' % \
        (len(freq_bf_bit_pos_dict))
  print '   ', auxiliary.get_memory_usage()
  print

  return freq_bf_bit_pos_dict


# -----------------------------------------------------------------------------

def gen_freq_bf_bit_positions2(encode_bf_bit_pos_list, min_count,
                               col_filter_set=set(),
                               row_filter_bit_array=None,
                               verbose=False):
  """Using an Apriori based approach, find all individual, pairs, triplets,
     etc. of bit positions that occur frequently together in the given list of
     bit position arrays (column-wise BFs).

     Only consider bit positions (and pairs and tuples of them) that have a
     Hamming weight of at least 'min_count'.

     If 'col_filter_set' is given (not an empty set), then do not consider
     columns listed in the set.

     If 'row_filter_bit_array' is given (not None), then do not consider the
     rows (BFs) that have a 0-bit.

     In this version of the function we do keep the actual conjunctions of BFs
     instead of only the set of bit positions, to check if the Apriori
     algorithm runs faster, andhow much more memory is needed.

     Return a dictionary where keys are the longest found tuples made of bit
     positions (integers) and values their counts of occurrences.
  """

  num_bf = len(encode_bf_bit_pos_list[0])

  # If needed generate the row filter bit array - set all rows (BFs) in the
  # filter set to 1 so all are considered
  #
  if (row_filter_bit_array == None):
    row_filter_bit_array = bitarray.bitarray(num_bf)
    row_filter_bit_array.setall(1)

  if (row_filter_bit_array != None):
    part_size = int(row_filter_bit_array.count(1))
  else:
    part_size = num_bf

  start_time = time.time()

  print 'Generate frequent bit position sets with HW of at least %d' % \
        (min_count)
  print '  Partiton size: %d Bfs (from %d total BFs)' % (part_size, num_bf)

  # First get all bit positions with a HW of at least 'min_count' - - - - - - -
  #
  freq_bit_pos_dict =    {}
  freq_bit_pos_hw_dict = {}  # And a dictionary where we keep their Hamming
                             # weights for printing

  ind_start_time = time.time()  # Time to get frequent individual positions

  max_count = -1

  for (pos, bit_array) in enumerate(encode_bf_bit_pos_list):

    # Only consider columns not given in the column filter set
    #
    if (pos not in col_filter_set):

      # Filter (AND) with row filter bit array
      #
      bit_pos_array_filtered = bit_array & row_filter_bit_array

      bit_pos_hw = int(bit_pos_array_filtered.count(1))
      max_count = max(max_count, bit_pos_hw)
      if (bit_pos_hw >= min_count):
        freq_bit_pos_dict[pos] =    bit_pos_array_filtered
        freq_bit_pos_hw_dict[pos] = bit_pos_hw

  print '  Found %d bit positions with a HW of at least %d (from %d BFs):' % \
        (len(freq_bit_pos_dict), min_count, num_bf)

  if (verbose == True):  # Print the 10 most and least frequent ones
    if (len(freq_bit_pos_dict) <= 20):
      for (pos, count) in sorted(freq_bit_pos_hw_dict.items(),
                                 key=lambda t: t[1], reverse=True):
        print '    %d: %d (%.2f%% / %.2f%%)' % \
              (pos, count, 100.0*count/num_bf, 100.0*count/part_size)
    else:
      for (pos, count) in sorted(freq_bit_pos_hw_dict.items(),
                                 key=lambda t: t[1], reverse=True)[:10]:
        print '    %d: %d (%.2f%% / %.2f%%)' % \
              (pos, count, 100.0*count/num_bf, 100.0*count/part_size)
      print '        ....'
      for (pos, count) in sorted(freq_bit_pos_hw_dict.items(),
                                 key=lambda t: t[1], reverse=True)[-10:]:
        print '    %d: %d (%.2f%% / %.2f%%)' % \
              (pos, count, 100.0*count/num_bf, 100.0*count/part_size)
    print

  print '  Generation of frequent individual BF bit positions took %.2f sec' \
        % (time.time()-ind_start_time)
  print '   ', auxiliary.get_memory_usage()
  print

  auxiliary.check_memory_use(MAX_MEMORY_USE)

  # Next get all pairs of bit positions with a HW of at least 'min_count' - -
  #
  freq_bit_pos_pair_dict =    {}
  freq_bit_pos_pair_hw_dict = {}  # Keep HW for printing

  pair_start_time = time.time()  # Time to get frequent pairs of positions

  freq_bit_pos_list = sorted(freq_bit_pos_dict.keys())

  for (i, pos1) in enumerate(freq_bit_pos_list[:-1]):
    bit_pos_bf1 = freq_bit_pos_dict[pos1]

    for pos2 in freq_bit_pos_list[i+1:]:
      assert pos1 < pos2, (pos1, pos2)

      # Get the bit-wise AND of the two position BFs
      #
      and_bit_pair_array = bit_pos_bf1 & freq_bit_pos_dict[pos2]

      and_bit_pos_pair_hw = int(and_bit_pair_array.count(1))

      if (and_bit_pos_pair_hw >= min_count):
        freq_bit_pos_pair_dict[(pos1,pos2)] = and_bit_pair_array
        freq_bit_pos_pair_hw_dict[(pos1,pos2)] = and_bit_pos_pair_hw

  # If no frequent pairs then return frequent individual bit positions and their
  # Hamming weights
  #
  if (len(freq_bit_pos_pair_dict) == 0):
    freq_bit_pos_hw_dict = {}  # Generate a dictionary of tuples and their HWs

    for (bit_pos, bit_pos_hw) in freq_bit_pos_hw_dict.iteritems():
      freq_bit_pos_hw_dict[(bit_pos,)] = bit_pos_hw

    return freq_bit_pos_hw_dict

  print '  Found %d bit position pairs with a HW of at least ' % \
        (len(freq_bit_pos_pair_dict)) + '%d (from %d BFs):' % \
        (min_count, num_bf)

  if (verbose == True):  # Print the 10 most and least frequent pairs
    if (len(freq_bit_pos_pair_dict) <= 20):
      for (pos_pair, count) in sorted(freq_bit_pos_pair_hw_dict.items(),
                                 key=lambda t: t[1], reverse=True):
        print '    %s: %d (%.2f%% / %.2f%%)' % \
              (str(pos_pair), count, 100.0*count/num_bf, 100.0*count/part_size)
    else:
      for (pos_pair, count) in sorted(freq_bit_pos_pair_hw_dict.items(),
                                 key=lambda t: t[1], reverse=True)[:10]:
        print '    %s: %d (%.2f%% / %.2f%%)' % \
              (str(pos_pair), count, 100.0*count/num_bf, 100.0*count/part_size)
      print '        ....'
      for (pos_pair, count) in sorted(freq_bit_pos_pair_hw_dict.items(),
                                      key=lambda t: t[1], reverse=True)[-10:]:
        print '    %s: %d (%.2f%% / %.2f%%)' % \
              (str(pos_pair), count, 100.0*count/num_bf, 100.0*count/part_size)
    print

  print '  Generation of frequent pairs of BF bit positions took %.2f sec' % \
        (time.time()-pair_start_time)
  print '   ', auxiliary.get_memory_usage()
  print

  auxiliary.check_memory_use(MAX_MEMORY_USE)

  prev_freq_bit_pos_tuple_dict = freq_bit_pos_pair_dict

  # If no frequent tuples of length 3 or more are found then return the pairs
  #
  freq_bf_bit_pos_dict = freq_bit_pos_pair_dict

  curr_len_m1 = 1  # Now run Apriori for sets of size 3 and more
  curr_len_p1 = 3

  while (len(prev_freq_bit_pos_tuple_dict) > 1):

    prev_freq_bit_pos_tuple_list = sorted(prev_freq_bit_pos_tuple_dict.keys())

#    print ' After sorting (line 1430) ', auxiliary.get_memory_usage()
#    print

    loop_start_time = time.time()

    # Generate candidates of current length plus 1
    #
    cand_bit_pos_tuple_dict =    {}
    cand_bit_pos_tuple_hw_dict = {}  # Keep HW for printing

    for (i, pos_tuple1) in enumerate(prev_freq_bit_pos_tuple_list[:-1]):

#      print ' With in Apriori (line 1442) ', auxiliary.get_memory_usage()
#      print

      pos_tuple1_m1 =   pos_tuple1[:curr_len_m1]
      pos_tuple1_last = pos_tuple1[-1]

      pos_tuple_bf1 = prev_freq_bit_pos_tuple_dict[pos_tuple1]

      for pos_tuple2 in prev_freq_bit_pos_tuple_list[i+1:]:

        # Check if the two tuples have the same beginning
        #
        if (pos_tuple1_m1 == pos_tuple2[:curr_len_m1]):
          assert pos_tuple1_last < pos_tuple2[-1], (pos_tuple1, pos_tuple2)
          cand_pos_tuple = pos_tuple1 + (pos_tuple2[-1],)

          # Check all sub-tuples are in previous frequent tuple set
          #
          all_sub_tuple_freq = True
          for pos in range(curr_len_p1):
            check_tuple = tuple(cand_pos_tuple[:pos] + \
                                cand_pos_tuple[pos+1:])
            if (check_tuple not in prev_freq_bit_pos_tuple_dict):
              all_sub_tuple_freq = False
              break

          if (all_sub_tuple_freq == True):  # Get intersection of positions

            and_bit_tuple_array = pos_tuple_bf1 & \
                                      prev_freq_bit_pos_tuple_dict[pos_tuple2]

            and_bit_pos_tuple_hw = int(and_bit_tuple_array.count(1))

            if (and_bit_pos_tuple_hw >= min_count):
              cand_bit_pos_tuple_dict[cand_pos_tuple] = and_bit_tuple_array
              cand_bit_pos_tuple_hw_dict[cand_pos_tuple] = and_bit_pos_tuple_hw

    if (len(cand_bit_pos_tuple_dict) == 0):
      break  # No more candidates, end Apriori process

    print '  Found %d bit position tuples of length %d with a HW of at ' % \
          (len(cand_bit_pos_tuple_dict), curr_len_p1) + \
          'least %d (from %d BFs):' % (min_count, num_bf)

    if (verbose == True):  # Print the 10 most and least frequent tuples
      if (len(cand_bit_pos_tuple_dict) <= 20):
        for (pos_tuple, count) in sorted(cand_bit_pos_tuple_hw_dict.items(),
                                         key=lambda t: t[1], reverse=True):
          print '    %s: %d (%.2f%% / %.2f%%)' % \
                (str(pos_tuple),count, 100.0*count/num_bf,
                 100.0*count/part_size)
      else:
        for (pos_tuple, count) in sorted(cand_bit_pos_tuple_hw_dict.items(),
                                         key=lambda t: t[1],
                                         reverse=True)[:10]:
          print '    %s: %d (%.2f%% / %.2f%%)' % \
                (str(pos_tuple),count, 100.0*count/num_bf,
                 100.0*count/part_size)
        print '        ....'
        for (pos_tuple, count) in sorted(cand_bit_pos_tuple_hw_dict.items(),
                                         key=lambda t: t[1],
                                         reverse=True)[-10:]:
          print '    %s: %d (%.2f%% / %.2f%%)' % \
                (str(pos_tuple),count, 100.0*count/num_bf,
                 100.0*count/part_size)
      print

    print '  Generation of frequent BF bit position tuples took %.2f sec' % \
          (time.time()-loop_start_time)
    print '   ', auxiliary.get_memory_usage()
    print

    auxiliary.check_memory_use(MAX_MEMORY_USE)

    # Set found frequent bit position tuples as final dictionary
    #
    freq_bf_bit_pos_dict = cand_bit_pos_tuple_dict

    curr_len_m1 += 1
    curr_len_p1 += 1

    prev_freq_bit_pos_tuple_dict = cand_bit_pos_tuple_dict

  freq_bf_bit_pos_hw_dict = {}

  for (bit_pos_tuple, bit_tuple_array) in freq_bf_bit_pos_dict.iteritems():
    freq_bf_bit_pos_hw_dict[bit_pos_tuple] = int(bit_tuple_array.count(1))

  print 'Overall generation of frequent BF bit position sets took %.1f sec' % \
        (time.time()-start_time)
  print '  Identified %d frequent bit position sets' % \
        (len(freq_bf_bit_pos_hw_dict))
  print '   ', auxiliary.get_memory_usage()
  print

  return freq_bf_bit_pos_hw_dict

# -----------------------------------------------------------------------------

def gen_freq_q_gram_bit_post_dict(q_gram_pos_assign_dict,
                                  true_q_gram_pos_map_dict):
  """Generate two dictionaries which for each identified frequent q-gram
     contain its bit positions (either all or only the correct ones) based on
     the given dictionary of positions and q-grams assigned to them.

     Returns two dictionaries, the first containing all bit positions per
     q-gram while the second only contains correct bit positions (based on the
     given 'true_q_gram_pos_map_dict').
  """

  all_identified_q_gram_pos_dict =  {}  # Keys are q-grams, values sets of pos.
  corr_identified_q_gram_pos_dict = {}  # Only correct positions
  num_pos_removed = 0                   # For the corrected dictionary

  for (pos, pos_q_gram_set) in q_gram_pos_assign_dict.iteritems():

    for q_gram in pos_q_gram_set:

      # Check if this is a correct position for this q-gram
      #
      if q_gram in true_q_gram_pos_map_dict.get(pos, set()):
        correct_pos = True
      else:
        correct_pos = False

      q_gram_pos_set = all_identified_q_gram_pos_dict.get(q_gram, set())
      q_gram_pos_set.add(pos)
      all_identified_q_gram_pos_dict[q_gram] = q_gram_pos_set

      if (correct_pos == True):
        q_gram_pos_set = corr_identified_q_gram_pos_dict.get(q_gram, set())
        q_gram_pos_set.add(pos)
        corr_identified_q_gram_pos_dict[q_gram] = q_gram_pos_set
      else:
        num_pos_removed += 1

  # Check each q-gram has at least one position in the correct only dictionary
  #
  for q_gram in corr_identified_q_gram_pos_dict.keys():
    if (len(corr_identified_q_gram_pos_dict[q_gram]) == 0):
      del corr_identified_q_gram_pos_dict[q_gram]
      print '*** Warning: Q-gram "%s" has no correct position, so it is ' % \
            (q_grams) + 'removed ***'

  print 'Converted assigned position / q-gram dictionary into a q-gram / ' + \
        'position dictionary'
  print '  Dictionary of all q-grams contains %d q-grams' % \
        (len(all_identified_q_gram_pos_dict))
  print '  Dictionary of correct q-grams contains %d q-grams ' % \
        (len(corr_identified_q_gram_pos_dict)) + \
        '(with %d wrong position assignments removed)' % (num_pos_removed)
  print

  return all_identified_q_gram_pos_dict, corr_identified_q_gram_pos_dict

# -----------------------------------------------------------------------------
# Functions for step 4: Re-identify attribute values based on frequent bit pos.
# -----------------------------------------------------------------------------

def re_identify_attr_val_setinter(bf_must_have_q_gram_dict,
                                  bf_cannot_have_q_gram_dict,
                                  plain_q_gram_attr_val_dict,
                                  encode_rec_val_dict, max_num_many=10,
                                  verbose=False):
  """Based on the given dictionaries of must have and cannot have q-grams per
     Bloom filter, and the given plain-text and encoded data set's attribute
     values (the latter being the true encoded values for a BF), re-identify
     attribute values from the set of plain-text values that could have been
     encoded in a BF.

     This method implements a simple set intersection approach that only finds
     those attribute values (possibly none) that contain all must have q-grams
     in a Bloom filter.

     Calculate and return the number of:
     - BFs with no guesses
     - BFs with more than 'max_num_many' guesses
     - BFs with 1-to-1 guesses
     - BFs with correct 1-to-1 guesses
     - BFs with partially matching 1-to-1 guesses
     - BFs with 1-to-many guesses
     - BFs with 1-to-many correct guesses
     - BFs with partially matching 1-to-many guesses

     - Accuracy of 1-to-1 partial matching values based on common tokens
     - Accuracy of 1-to-many partial matching values based on common tokens

     Also returns a dictionary with BFs as keys and correctly re-identified
     attribute values as values.
  """

  print 'Re-identify encoded attribute values based on must have and ' + \
        'cannot have q-grams using set-intersections:'

  start_time = time.time()

  num_no_guess =       0
  num_too_many_guess = 0
  num_1_1_guess =      0
  num_corr_1_1_guess = 0
  num_part_1_1_guess = 0
  num_1_m_guess =      0
  num_corr_1_m_guess = 0
  num_part_1_m_guess = 0

  acc_part_1_1_guess = 0.0  # Average accuracy of partial matching values based
  acc_part_1_m_guess = 0.0  # on common tokens

  # BFs with correctly re-identified attribute values
  #
  corr_reid_attr_val_dict = {}

  rec_num = 0

  for (enc_rec_id, bf_q_gram_set) in bf_must_have_q_gram_dict.iteritems():

    st = time.time()

    reid_attr_set_list = []

    for q_gram in bf_q_gram_set:
      reid_attr_set_list.append(plain_q_gram_attr_val_dict[q_gram])

    reid_attr_set_list.sort(key=len)  # Shortest first so smaller intersections

    reid_attr_val_set = set.intersection(*reid_attr_set_list)

    # Remove the attribute values that contain must not have q-grams
    #
    if ((len(reid_attr_val_set) > 0) and \
        (enc_rec_id in bf_cannot_have_q_gram_dict)):
      must_not_have_q_gram_set = bf_cannot_have_q_gram_dict[enc_rec_id]

      checked_reid_attr_val_set = set()

      for attr_val in reid_attr_val_set:
        no_cannot_have_q_gram = True
        for q_gram in must_not_have_q_gram_set:
          if (q_gram in attr_val):
            no_cannot_have_q_gram = False
            break
        if (no_cannot_have_q_gram == True):
          checked_reid_attr_val_set.add(attr_val)

      reid_attr_val_set = checked_reid_attr_val_set

    num_bf_attr_val_guess = len(reid_attr_val_set)

    # Check if there are possible plain-text values for this BF
    #
    if (num_bf_attr_val_guess == 0):
      num_no_guess += 1
    elif (num_bf_attr_val_guess == 1):
      num_1_1_guess += 1
    elif (num_bf_attr_val_guess > max_num_many):
      num_too_many_guess += 1
    else:
      num_1_m_guess += 1

    # If there is a small number (<= max_num_many) of possible values check if
    # the correct one is included
    #
    if (num_bf_attr_val_guess >= 1) and (num_bf_attr_val_guess <= max_num_many):

      true_encoded_attr_val = encode_rec_val_dict[enc_rec_id]

      if (true_encoded_attr_val in reid_attr_val_set):

        # True attribute value is re-identified
        #
        corr_reid_attr_val_dict[enc_rec_id] = reid_attr_val_set

        if (num_bf_attr_val_guess == 1):
          num_corr_1_1_guess += 1
        else:
          num_corr_1_m_guess += 1

      else:  # If no exact match, check if some words / tokens are in common

        true_encoded_attr_val_set = set(true_encoded_attr_val.split())

        # Get maximum number of tokens shared with an encoded attribute value
        #
        max_num_common_token = 0

        for plain_text_attr_val in reid_attr_val_set:
          plain_text_attr_val_set = set(plain_text_attr_val.split())

          num_common_token = \
                       len(true_encoded_attr_val_set & plain_text_attr_val_set)
          max_num_common_token = max(max_num_common_token, num_common_token)

        if (max_num_common_token > 0):  # Add partial accuracy of common tokens
          num_token_acc = float(max_num_common_token) / \
                          len(true_encoded_attr_val_set)

          if (num_bf_attr_val_guess == 1):
            num_part_1_1_guess += 1
            acc_part_1_1_guess += num_token_acc
          else:
            num_part_1_m_guess += 1
            acc_part_1_m_guess += num_token_acc

    rec_num += 1

    if ((rec_num % 10000) == 0):  # Print intermediate result

      if (verbose == False):
        print '  Number of records processed: %d of %d' % \
               (rec_num, len(bf_must_have_q_gram_dict)) + \
               ' (in %.1f sec)' % (time.time() - start_time)
      else:
        print
        print '  Number of records processed: %d of %d' % \
               (rec_num, len(bf_must_have_q_gram_dict)) + \
               ' (in %.1f sec)' % (time.time() - start_time)

        print '    Num no guesses:                          %d' % \
              (num_no_guess)
        print '    Num > %d guesses:                        %d' % \
              (max_num_many, num_too_many_guess)
        print '    Num 2 to %d guesses:                     %d' % \
              (max_num_many, num_1_m_guess)
        print '      Num correct 2 to %d guesses:           %d' % \
              (max_num_many, num_corr_1_m_guess)
        if (num_part_1_m_guess > 0):
          print '      Num partially correct 2 to %d guesses: %d' % \
                (max_num_many, num_part_1_m_guess) + \
                ' (average accuracy of common tokens: %.2f)' % \
                (acc_part_1_m_guess / num_part_1_m_guess)
        print '    Num 1-1 guesses:                         %d' % \
              (num_1_1_guess)
        print '      Num correct 1-1 guesses:               %d' % \
              (num_corr_1_1_guess)
        if (num_part_1_1_guess > 0):
          print '      Num partially correct 1-1 guesses:     %d' % \
                (num_part_1_1_guess) + \
                ' (average accuracy of common tokens: %.2f)' % \
                (acc_part_1_1_guess / num_part_1_1_guess)

  total_time = time.time() - start_time

  if (num_part_1_m_guess > 0):
    acc_part_1_m_guess = float(acc_part_1_m_guess) / num_part_1_m_guess
  else:
    acc_part_1_m_guess = 0.0
  if (num_part_1_1_guess > 0):
    acc_part_1_1_guess = float(acc_part_1_1_guess) / num_part_1_1_guess
  else:
    acc_part_1_1_guess = 0.0

  print '  Total time required to re-identify from %d Bloom filters: ' % \
        (len(bf_must_have_q_gram_dict)) + '%.1f sec (%.2f msec per BF)' % \
        (total_time, 1000.0*total_time / len(bf_must_have_q_gram_dict))
  print
  print '  Num no guesses:                          %d' % (num_no_guess)
  print '  Num > %d guesses:                        %d' % \
        (max_num_many, num_too_many_guess)
  print '  Num 2 to %d guesses:                     %d' % \
        (max_num_many, num_1_m_guess)
  print '    Num correct 2 to %d guesses:           %d' % \
        (max_num_many, num_corr_1_m_guess)
  if (num_part_1_m_guess > 0):
    print '    Num partially correct 2 to %d guesses: %d' % \
          (max_num_many, num_part_1_m_guess) + \
          ' (average accuracy of common tokens: %.2f)' % (acc_part_1_m_guess)
  else:
    print '    No partially correct 2 to %d guesses' % (max_num_many)
  print '  Num 1-1 guesses:                         %d' % \
        (num_1_1_guess)
  print '    Num correct 1-1 guesses:               %d' % \
        (num_corr_1_1_guess)
  if (num_part_1_1_guess > 0):
    print '    Num partially correct 1-1 guesses:     %d' % \
          (num_part_1_1_guess) + \
          ' (average accuracy of common tokens: %.2f)' % (acc_part_1_1_guess)
  else:
    print '    No partially correct 1-1 guesses'
  print

  return num_no_guess, num_too_many_guess, num_1_1_guess, num_corr_1_1_guess, \
         num_part_1_1_guess, num_1_m_guess, num_corr_1_m_guess, \
         num_part_1_m_guess, acc_part_1_1_guess, acc_part_1_m_guess, \
         corr_reid_attr_val_dict

# -----------------------------------------------------------------------------

def re_identify_attr_val_apriori(bf_must_have_q_gram_dict,
                                 bf_cannot_have_q_gram_dict,
                                 plain_q_gram_attr_val_dict,
                                 encode_rec_val_dict,
                                 max_num_many=10, verbose=False):
  """Based on the given dictionaries of must have and cannot have q-grams per
     Bloom filter, and the given plain-text and encoded data set's attribute
     values (the latter being the true encoded values for a BF), re-identify
     attribute values from the set of plain-text values that could have been
     encoded in a BF.

     This method implements an Apriori based approach that finds those
     attribute values (possibly none) that contain most of the must have
     q-grams in a Bloom filter.

     Calculate and return the number of:
     - BFs with no guesses
     - BFs with more than 'max_num_many' guesses
     - BFs with 1-to-1 guesses
     - BFs with correct 1-to-1 guesses
     - BFs with partially matching 1-to-1 guesses
     - BFs with 1-to-many guesses
     - BFs with 1-to-many correct guesses
     - BFs with partially matching 1-to-many guesses

     - Accuracy of 1-to-1 partial matching values based on common tokens
     - Accuracy of 1-to-many partial matching values based on common tokens

     Also returns a dictionary with BFs as keys and correctly re-identified
     attribute values as values.
  """

  print 'Re-identify encoded attribute values based on must have and ' + \
        'cannot have q-grams using Apriori approach:'

  start_time = time.time()

  num_no_guess =       0
  num_too_many_guess = 0
  num_1_1_guess =      0
  num_corr_1_1_guess = 0
  num_part_1_1_guess = 0
  num_1_m_guess =      0
  num_corr_1_m_guess = 0
  num_part_1_m_guess = 0

  acc_part_1_1_guess = 0.0  # Average accuracy of partial matching values based
  acc_part_1_m_guess = 0.0  # on common tokens

  # BFs with correctly re-identified attribute values
  #
  corr_reid_attr_val_dict = {}

  rec_num = 0

  # Loop over those Bloom filters for which we have identified q-grams which
  # are believed to be encoded in the Bloom filter
  #
  for (rec_id, must_have_q_gram_set) in bf_must_have_q_gram_dict.iteritems():
    cannot_have_q_gram_set = bf_cannot_have_q_gram_dict.get(rec_id, set())

    assert len(must_have_q_gram_set & cannot_have_q_gram_set) == 0

    # Step 1: Get all q-gram pairs that occur in attribute values
    #
    q_gram_pair_attr_val_dict = {}

    must_have_q_gram_list = sorted(must_have_q_gram_set)

    for (i, q_gram1) in enumerate(must_have_q_gram_list[:-1]):
      q_gram_attr_val_set1 = plain_q_gram_attr_val_dict[q_gram1]

      for q_gram2 in must_have_q_gram_list[i+1:]:

        # Intersection of attribute value sets to get those with both q-grams
        #
        q_gram_pair_attr_val_set = q_gram_attr_val_set1 & \
                                   plain_q_gram_attr_val_dict[q_gram2]

        if (len(q_gram_pair_attr_val_set) > 0):

          # Remove attribute values that contain any cannot have q-grams
          #
          for attr_val in list(q_gram_pair_attr_val_set):
            for q_gram in cannot_have_q_gram_set:
              if (q_gram in attr_val):
                q_gram_pair_attr_val_set.remove(attr_val)
                break

          if (len(q_gram_pair_attr_val_set) > 0):
            q_gram_pair_attr_val_dict[(q_gram1,q_gram2)] = \
                                                    q_gram_pair_attr_val_set

    if (len(q_gram_pair_attr_val_dict) == 0):
      reid_attr_val_set = set()  # Don't just consider individual q-grams

    else:  # There are pairs of q-grams with attribute values

      # Step 2: Run Apriori to get longer tuples of q-grams and their attribute
      #         values
      #
      curr_len_m1 = 1
      curr_len_p1 = 3

      prev_q_gram_tuple_attr_val_dict = q_gram_pair_attr_val_dict

      while (len(prev_q_gram_tuple_attr_val_dict) > 1):

        prev_q_gram_tuple_list = sorted(prev_q_gram_tuple_attr_val_dict.keys())

        # Generate candidates of current length plus 1
        #
        cand_q_gram_tuple_attr_val_dict = {}

        for (i, q_gram_tuple1) in enumerate(prev_q_gram_tuple_list[:-1]):

          q_gram_tuple1_m1 =   q_gram_tuple1[:curr_len_m1]
          q_gram_tuple1_last = q_gram_tuple1[-1]

          q_gram_tuple_attr_val_set1 = \
                             prev_q_gram_tuple_attr_val_dict[q_gram_tuple1]

          for q_gram_tuple2 in prev_q_gram_tuple_list[i+1:]:

            # Check if the two tuples have the same beginning
            #
            if (q_gram_tuple1_m1 == q_gram_tuple2[:curr_len_m1]):
              assert q_gram_tuple1_last < q_gram_tuple2[-1], \
                     (q_gram_tuple1, q_gram_tuple2)
              cand_q_gram_tuple = q_gram_tuple1 + (q_gram_tuple2[-1],)

              # Check all sub-tuples are in previous frequent tuple set
              #
              all_sub_tuple_freq = True
              for pos in range(curr_len_p1):
                check_tuple = tuple(cand_q_gram_tuple[:pos] + \
                                    cand_q_gram_tuple[pos+1:])
                if (check_tuple not in prev_q_gram_tuple_attr_val_dict):
                  all_sub_tuple_freq = False
                  break

              if (all_sub_tuple_freq == True):  # Get intersection of attr vals
                cand_q_gram_tuple_attr_val_set = q_gram_tuple_attr_val_set1 & \
                             prev_q_gram_tuple_attr_val_dict[q_gram_tuple2]

                if (len(cand_q_gram_tuple_attr_val_set) > 0):
                  cand_q_gram_tuple_attr_val_dict[cand_q_gram_tuple] = \
                           cand_q_gram_tuple_attr_val_set

        if (len(cand_q_gram_tuple_attr_val_dict) == 0):
          break  # No more candidates, end Apriori process

        prev_q_gram_tuple_attr_val_dict = cand_q_gram_tuple_attr_val_dict

        curr_len_m1 += 1
        curr_len_p1 += 1

      # Set found attribute values as the union of all re-identified values of
      # q-gram tuples possible for this BF
      #
      reid_attr_val_set = set()
      for attr_val_set in prev_q_gram_tuple_attr_val_dict.itervalues():
        reid_attr_val_set.update(attr_val_set)

    num_bf_attr_val_guess = len(reid_attr_val_set)

    # Check if there are possible plain-text values for this BF
    #
    if (num_bf_attr_val_guess == 0):
      num_no_guess += 1
    elif (num_bf_attr_val_guess == 1):
      num_1_1_guess += 1
    elif (num_bf_attr_val_guess > max_num_many):
      num_too_many_guess += 1
    else:
      num_1_m_guess += 1

    # If there is a small number (<= max_num_many) of possible values check if
    # the correct one is included
    #
    if (num_bf_attr_val_guess >= 1) and (num_bf_attr_val_guess <= max_num_many):

      true_encoded_attr_val = encode_rec_val_dict[rec_id]

      if (true_encoded_attr_val in reid_attr_val_set):

        # True attribute value is re-identified
        #
        corr_reid_attr_val_dict[rec_id] = reid_attr_val_set

        if (num_bf_attr_val_guess == 1):
          num_corr_1_1_guess += 1
        else:
          num_corr_1_m_guess += 1

      else:  # If no exact match, check if some words / tokens are in common

        true_encoded_attr_val_set = set(true_encoded_attr_val.split())

        # Get maximum number of tokens shared with an encoded attribute value
        #
        max_num_common_token = 0

        for plain_text_attr_val in reid_attr_val_set:
          plain_text_attr_val_set = set(plain_text_attr_val.split())

          num_common_token = \
                       len(true_encoded_attr_val_set & plain_text_attr_val_set)
          max_num_common_token = max(max_num_common_token, num_common_token)

        if (max_num_common_token > 0):  # Add partial accuracy of common tokens
          num_token_acc = float(max_num_common_token) / \
                          len(true_encoded_attr_val_set)

          if (num_bf_attr_val_guess == 1):
            num_part_1_1_guess += 1
            acc_part_1_1_guess += num_token_acc
          else:
            num_part_1_m_guess += 1
            acc_part_1_m_guess += num_token_acc

    rec_num += 1

    if ((rec_num % 10000) == 0):  # Print intermediate result

      if (verbose == False):
        print '  Number of records processed: %d of %d' % \
               (rec_num, len(bf_must_have_q_gram_dict)) + \
               ' (in %.1f sec)' % (time.time() - start_time)
      else:
        print
        print '  Number of records processed: %d of %d' % \
               (rec_num, len(bf_must_have_q_gram_dict)) + \
               ' (in %.1f sec)' % (time.time() - start_time)

        print '    Num no guesses:                          %d' % \
              (num_no_guess)
        print '    Num > %d guesses:                        %d' % \
              (max_num_many, num_too_many_guess)
        print '    Num 2 to %d guesses:                     %d' % \
              (max_num_many, num_1_m_guess)
        print '      Num correct 2 to %d guesses:           %d' % \
              (max_num_many, num_corr_1_m_guess)
        if (num_part_1_m_guess > 0):
          print '      Num partially correct 2 to %d guesses: %d' % \
                (max_num_many, num_part_1_m_guess) + \
                ' (average accuracy of common tokens: %.2f)' % \
                (acc_part_1_m_guess / num_part_1_m_guess)
        print '    Num 1-1 guesses:                         %d' % \
              (num_1_1_guess)
        print '      Num correct 1-1 guesses:               %d' % \
              (num_corr_1_1_guess)
        if (num_part_1_1_guess > 0):
          print '      Num partially correct 1-1 guesses:     %d' % \
                (num_part_1_1_guess) + \
                ' (average accuracy of common tokens: %.2f)' % \
                (acc_part_1_1_guess / num_part_1_1_guess)

  total_time = time.time() - start_time

  if (num_part_1_m_guess > 0):
    acc_part_1_m_guess = float(acc_part_1_m_guess) / num_part_1_m_guess
  else:
    acc_part_1_m_guess = 0.0
  if (num_part_1_1_guess > 0):
    acc_part_1_1_guess = float(acc_part_1_1_guess) / num_part_1_1_guess
  else:
    acc_part_1_1_guess = 0.0

  print '  Total time required to re-identify from %d Bloom filters: ' % \
        (len(bf_must_have_q_gram_dict)) + '%.1f sec (%.2f msec per BF)' % \
        (total_time, 1000.0*total_time / len(bf_must_have_q_gram_dict))
  print
  print '  Num no guesses:                          %d' % (num_no_guess)
  print '  Num > %d guesses:                        %d' % \
        (max_num_many, num_too_many_guess)
  print '  Num 2 to %d guesses:                     %d' % \
        (max_num_many, num_1_m_guess)
  print '    Num correct 2 to %d guesses:           %d' % \
        (max_num_many, num_corr_1_m_guess)
  if (num_part_1_m_guess > 0):
    print '    Num partially correct 2 to %d guesses: %d' % \
          (max_num_many, num_part_1_m_guess) + \
          ' (average accuracy of common tokens: %.2f)' % (acc_part_1_m_guess)
  else:
    print '    No partially correct 2 to %d guesses' % (max_num_many)
  print '  Num 1-1 guesses:                         %d' % \
        (num_1_1_guess)
  print '    Num correct 1-1 guesses:               %d' % \
        (num_corr_1_1_guess)
  if (num_part_1_1_guess > 0):
    print '    Num partially correct 1-1 guesses:     %d' % \
          (num_part_1_1_guess) + \
          ' (average accuracy of common tokens: %.2f)' % (acc_part_1_1_guess)
  else:
    print '    No partially correct 1-1 guesses'
  print

  return num_no_guess, num_too_many_guess, num_1_1_guess, num_corr_1_1_guess, \
         num_part_1_1_guess, num_1_m_guess, num_corr_1_m_guess, \
         num_part_1_m_guess, acc_part_1_1_guess, acc_part_1_m_guess, \
         corr_reid_attr_val_dict

# -----------------------------------------------------------------------------

def get_matching_q_gram_sets(identified_q_gram_pos_dict, encode_bf_dict,
                             plain_q_gram_attr_val_dict,
                             plain_attr_val_rec_id_dict,
                             bf_cannot_have_q_gram_dict, bf_len):
  """This function finds for each pattern of one or more frequent q-grams the
     sets of plain-text attribute values that contain these q-grams, and the
     corresponding sets of BFs that potentially could encode these attribute
     values.

     The function returns a dictionary where frequent q-gram tuples are keys,
     and values are two sets of record identifiers, one of encoded BFs that
     have matching 1-bits in all relevant positions for the q-grams in the
     key, and the second with record identifiers from the plain-text data set
     that contain all the q-grams in the key.
  """

  start_time = time.time()

  # The dictionary to be returned with q-gram tuples as keys and two sets of
  # record identifiers (one corresponding to encoded BFs, the other to
  # plain-text values) for each such q-gram tuple.
  #
  q_gram_tuple_rec_id_dict = {}

  # The list of frequent q-grams we have
  #
  freq_q_gram_list = sorted(identified_q_gram_pos_dict.keys())

  print 'Find q-gram tuples that have corresponding BFs and attribute ' + \
        'values:'
  print '  %d frequent q-grams:' % (len(freq_q_gram_list)), freq_q_gram_list
  print

  # Step 1: For each pair of frequent q-grams, get the attribute values from
  #         the plain-text data set that contain both q-grams
  #
  q_gram_pair_attr_val_rec_id_dict = {}

  # Count the number of q-grams pairs with no attribute values that contain
  # both q-grams
  #
  num_q_gram_pair_not_occurring = 0

  num_attr_val_pair_list = []  # To calculate statistics

  for (i, freq_q_gram1) in enumerate(freq_q_gram_list[:-1]):

    q_gram_attr_val_set1 = plain_q_gram_attr_val_dict[freq_q_gram1]

    for freq_q_gram2 in freq_q_gram_list[i+1:]:

      # Get the set of attribute values that contain both q-grams
      #
      common_attr_val_set = q_gram_attr_val_set1 & \
                            plain_q_gram_attr_val_dict[freq_q_gram2]

      if (len(common_attr_val_set) > 0):
        q_gram_pair_attr_val_rec_id_dict[(freq_q_gram1,freq_q_gram2)] = \
           common_attr_val_set
        q_gram_tuple_rec_id_dict[(freq_q_gram1,freq_q_gram2)] = \
           common_attr_val_set  # Add to final dictionary to be returned

        num_attr_val_pair_list.append(len(common_attr_val_set))

      else:
        num_q_gram_pair_not_occurring += 1

  print '  %d q-gram pairs occur in plain-text attribute values' % \
        (len(q_gram_pair_attr_val_rec_id_dict))
  if (len(q_gram_pair_attr_val_rec_id_dict) > 0):
    print '    Minimum, average and maximum number of attribute values per ' \
          + 'frequent q-gram pair: %d / %.2f / %d' % \
          (min(num_attr_val_pair_list), numpy.mean(num_attr_val_pair_list),
          max(num_attr_val_pair_list))
  print '    %d pairs of q-grams had no plain-text attribute values' % \
        (num_q_gram_pair_not_occurring)
  print

  # Step 2: Now run Apriori for sets of size 3 and more
  #
  curr_len_m1 = 1
  curr_len =    2
  curr_len_p1 = 3

  prev_q_gram_tuple_attr_val_rec_id_dict = q_gram_pair_attr_val_rec_id_dict

  while (len(prev_q_gram_tuple_attr_val_rec_id_dict) > 1):
    loop_start_time = time.time()

    # Get the list of all previous q-gram tuples (of length 'curr_len')
    #
    prev_q_gram_tuple_list = \
                         sorted(prev_q_gram_tuple_attr_val_rec_id_dict.keys())

    # Generate candidates of current length plus 1
    #
    cand_q_gram_tuple_attr_val_rec_id_dict = {}

    for (i, q_gram_tuple1) in enumerate(prev_q_gram_tuple_list[:-1]):

      q_gram_tuple1_m1 =   q_gram_tuple1[:curr_len_m1]
      q_gram_tuple1_last = q_gram_tuple1[-1]

      # Get the attribute value set of the first previous q-gram tuple
      #
      prev_tuple_attr_val_set1 = \
                 prev_q_gram_tuple_attr_val_rec_id_dict[q_gram_tuple1]

      for q_gram_tuple2 in prev_q_gram_tuple_list[i+1:]:

        # Check if the two tuples have the same beginning
        #
        if (q_gram_tuple1_m1 == q_gram_tuple2[:curr_len_m1]):
          assert q_gram_tuple1_last < q_gram_tuple2[-1], \
                 (q_gram_tuple1, q_gram_tuple2)
          cand_q_gram_tuple = q_gram_tuple1 + (q_gram_tuple2[-1],)

          # Check all sub-tuples are in previous tuple set
          #
          all_sub_tuple_freq = True
          for j in range(curr_len_p1):
            check_tuple = tuple(cand_q_gram_tuple[:j] + \
                                cand_q_gram_tuple[j+1:])
            if (check_tuple not in prev_q_gram_tuple_attr_val_rec_id_dict):
              all_sub_tuple_freq = False
              break

          # If all sub-sets are in previous dictionary, check if there are
          # plain-text attribute values that contain all q-grams in the
          # candidate tuple
          #
          if (all_sub_tuple_freq == True):

            cand_q_gram_tuple_attr_val_set = prev_tuple_attr_val_set1 & \
                        prev_q_gram_tuple_attr_val_rec_id_dict[q_gram_tuple2]

            if (len(cand_q_gram_tuple_attr_val_set) > 0):
              cand_q_gram_tuple_attr_val_rec_id_dict[cand_q_gram_tuple] = \
                                                cand_q_gram_tuple_attr_val_set

    if (len(cand_q_gram_tuple_attr_val_rec_id_dict) == 0):
      break  # No more candidates, end Apriori process

    # Remove all shorter q-gram tuples that are true sub-sets of one of the
    # new longer tuples
    #
    num_del_shorter_tuple = 0

    for long_q_gram_tuple in cand_q_gram_tuple_attr_val_rec_id_dict.iterkeys():
      long_q_gram_tuple_set = set(long_q_gram_tuple)
      assert len(long_q_gram_tuple_set) == curr_len_p1, long_q_gram_tuple_set

      for short_q_gram_tuple in q_gram_tuple_rec_id_dict.keys():
        if (len(short_q_gram_tuple) == curr_len):
          if (set(short_q_gram_tuple).issubset(long_q_gram_tuple_set)):
            del q_gram_tuple_rec_id_dict[short_q_gram_tuple]
            num_del_shorter_tuple += 1

    print '    Removed %d q-gram tuples of length %d that are true sub-sets' \
          % (num_del_shorter_tuple, curr_len)+' of q-gram tuples of length', \
          curr_len_p1
    print

    # ************************************************************************
    # PC TODO 20170925: the above code is quadratic - a better way? During the
    # generation of longer tuples?
    # ************************************************************************

    # Add to the final dictionary to be returned
    #
    for (q_gram_tuple, attr_val_set) in \
                     cand_q_gram_tuple_attr_val_rec_id_dict.iteritems():
      q_gram_tuple_rec_id_dict[q_gram_tuple] = attr_val_set

    num_attr_val_list = []  # Calculate statistics of sets

    for attr_val_set in cand_q_gram_tuple_attr_val_rec_id_dict.itervalues():
      num_attr_val_list.append(len(attr_val_set))

    print '    %d q-gram tuples of size %d had plain-text attribute ' % \
          (len(cand_q_gram_tuple_attr_val_rec_id_dict), curr_len_p1) + \
          'values'
    if (len(num_attr_val_list) > 0):
      print '      Minimum, average and maximum number of attribute values' + \
            ' per q-gram tuple BF: %d / %.2f / %d' % \
            (min(num_attr_val_list), numpy.mean(num_attr_val_list),
             max(num_attr_val_list))
    print

    auxiliary.check_memory_use(MAX_MEMORY_USE)

    prev_q_gram_tuple_attr_val_rec_id_dict = \
                                    cand_q_gram_tuple_attr_val_rec_id_dict
    curr_len_m1 += 1
    curr_len +=    1
    curr_len_p1 += 1

  num_attr_val_list = []
  for attr_val_set in q_gram_tuple_rec_id_dict.itervalues():
    num_attr_val_list.append(len(attr_val_set))

  num_q_gram_plain_tuples = len(q_gram_tuple_rec_id_dict)

  print '  Identified %d q-gram tuples that have plain-text attribute ' % \
        (num_q_gram_plain_tuples) + 'values'
  if (num_q_gram_plain_tuples > 0):
    print '    Minimum, average and maximum number of attribute values ' + \
          'per q-gram tuple BF: %d / %.2f / %d' % \
          (min(num_attr_val_list), numpy.mean(num_attr_val_list),
           max(num_attr_val_list))
  print

  # Step 3: For each identified q-gram tuple, get the corresponding BF 1-bit
  #         filter and check if there are encoded BFs that have the
  #         corresponding bit pattern
  #
  num_q_gram_tuple_removed = 0  # Count how many tuples have no matching BF

  print '  Get matching BF bit patterns for each q-gram tuple:'

  for (i,q_gram_tuple) in enumerate(sorted(q_gram_tuple_rec_id_dict.keys())):
    if ((i+1) % 100 == 0):
      print '    Processed %d of %d tuples' % \
            (i+1, num_q_gram_plain_tuples)

    # Generate the BF filter (1-bits in all positions where frequent q-grams
    # require them)
    #
    q_gram_bf_filter = bitarray.bitarray(bf_len)
    q_gram_bf_filter.setall(0)

    # Set 1-bits for all frequent q-grams in this tuple
    #
    for q_gram in q_gram_tuple:
      for pos in identified_q_gram_pos_dict[q_gram]:
        q_gram_bf_filter[pos] = 1

    # Get all record identifiers from the encoded data set that potentially can
    # encode this q-gram tuple
    #
    q_gram_tuple_bf_rec_id_set = set()

    # Loop over all BFs in the encoded data set
    #
    for (enc_rec_id, rec_bf) in encode_bf_dict.iteritems():

      # All 1-bits in the filter must also be set in a record BF
      #
      if (rec_bf & q_gram_bf_filter == q_gram_bf_filter):
        q_gram_tuple_bf_rec_id_set.add(enc_rec_id)

    # If there are BFs (encoded records) then get the plain-text record
    # identifiers of the attribute values identified in step 2
    #
    if (len(q_gram_tuple_bf_rec_id_set) > 0):

      # Plain-text record identifiers with these attribute values
      #
      q_gram_tuple_attr_val_rec_id_set = set()

      # Add attribute values that contain all q-grams in the tuple
      #
      for attr_val in q_gram_tuple_rec_id_dict[q_gram_tuple]:

        for plain_rec_id in plain_attr_val_rec_id_dict[attr_val]:
          q_gram_tuple_attr_val_rec_id_set.add(plain_rec_id)

      # If there are records with this attribute value then add both the BF
      # and attribute value record identifier sets to the final dictionary
      #
      if (len(q_gram_tuple_attr_val_rec_id_set) > 0):

        q_gram_tuple_rec_id_dict[q_gram_tuple] = \
                         (q_gram_tuple_bf_rec_id_set,
                          q_gram_tuple_attr_val_rec_id_set)
     
      else:  # No attribute values so delete this q-gram tuple
        num_q_gram_tuple_removed += 1
        del q_gram_tuple_rec_id_dict[q_gram_tuple]

        # This should not really happen ******************************
        print '*** Warning - this should not happen ***', \
              q_gram_tuple_attr_val_rec_id_set # ******************

    # If no BFs then delete this q-gram tuple from q_gram_tuple_rec_id_dict
    #
    else:
      num_q_gram_tuple_removed += 1
      del q_gram_tuple_rec_id_dict[q_gram_tuple]

  # Now check if we can remove plain-text values (records) because they contain
  # cannot have q-grams
  #
  checked_q_gram_tuple_rec_id_dict = {}

  for q_gram_tuple in q_gram_tuple_rec_id_dict:
    enc_rec_id_set =   q_gram_tuple_rec_id_dict[q_gram_tuple][0]
    plain_rec_id_set = q_gram_tuple_rec_id_dict[q_gram_tuple][1]

    # For each encoded record check each plain-text record
    #
    enc_rec_plain_rec_id_set = set()

    for enc_rec_id in enc_rec_id_set:
      cannot_q_gram_set = bf_cannot_have_q_gram_dict.get(enc_rec_id, set())

      # Check if a plain-text record contains cannot have q-grams
      #
      for plain_rec_id in plain_rec_id_set:
        plain_rec_q_gram_set = plain_q_gram_dict[plain_rec_id]

        no_cannot_q_gram = True
        for q_gram in cannot_q_gram_set:
          if q_gram in plain_rec_q_gram_set:
            no_cannot_q_gram = False
            break

        if (no_cannot_q_gram == True):
          enc_rec_plain_rec_id_set.add(plain_rec_id)

      if (len(enc_rec_plain_rec_id_set) > 0):  # There are plain-text records

        # Generate a new unique key for this encoded BF
        #
        checked_q_gram_tuple = q_gram_tuple + (enc_rec_id,)

        checked_q_gram_tuple_rec_id_dict[checked_q_gram_tuple] = \
                      (set([enc_rec_id]), enc_rec_plain_rec_id_set)

  print
  print '  Identified %d q-gram tuples that match both encoded BFs and ' % \
        (len(checked_q_gram_tuple_rec_id_dict)) + 'plain-text attribute values'
  print '    Removed %d q-gram tuples that had no encoded BFs' % \
        (num_q_gram_tuple_removed)
  print '    Overall matching of BFs and attribute values took %.1f sec' % \
        (time.time()-start_time)
  print '   ', auxiliary.get_memory_usage()
  print

  return checked_q_gram_tuple_rec_id_dict

# -----------------------------------------------------------------------------

def get_matching_bf_q_gram_sets(identified_q_gram_pos_dict, encode_bf_dict,
                                plain_q_gram_dict, bf_cannot_have_q_gram_dict,
                                bf_len):
  """Based on the given identified bit position tuples of frequent q-grams, the
     given BF dictionary (assumed to come from the encoded data set), and the
     given q-gram dictionary (assumed to contain the q-grams from records in
     the plain-text data set), for each tuple of frequent q-grams identify all
     pairs of BFs / q-gram sets that both contain these frequent q-grams as
     well as have 1-bits in all corresponding bit positions.

     Only keep q-gram tuples that are not sub-sets of longer q-gram tuples.

     The function returns a dictionary where frequent q-gram tuples are keys,
     and values are two sets of record identifiers, one of encoded BFs that
     have matching 1-bits in all relevant positions for the q-grams in the
     key, and the second with record identifiers from the plain-text data set
     that contain all the q-grams in the key.
  """

  start_time = time.time()

  # The dictionary to be returned with q-gram tuples as keys and two sets of
  # record identifiers (one corresponding to encoded BFs, the other to
  # plain-text values) for each such q-gram tuples.
  #
  q_gram_tuple_rec_id_dict = {}

  # The list of frequent q-grams we have
  #
  freq_q_gram_list = sorted(identified_q_gram_pos_dict.keys())

  print 'Find q-gram tuples that have corresponding BFs and attribute ' + \
        'values:'
  print '  %d frequent q-grams:' % (len(freq_q_gram_list)), freq_q_gram_list
  print

  # Step 1: For each pair of frequent q-grams and the union of their 1-bit
  #         positions, get the record identifiers that have the corresponding
  #         1-bit positions, and the plain-text values that contain both
  #         q-grams
  #
  q_gram_pair_bf_rec_id_dict =     {}
  q_gram_pair_q_gram_rec_id_dict = {}

  q_gram_pair_bf_dict = {}  # Also generate the BF for this q-gram pair

  # Count the number of q-grams pairs with either no BF in the encoded data set
  # and/or no plain-text attribute value that contains both q-grams
  #
  num_q_gram_pair_not_occurring = 0

  for (i, freq_q_gram1) in enumerate(freq_q_gram_list[:-1]):

    # Generate the BF for first q-gram with 1-bits in all required positions
    # 
    q_gram_bf1 = bitarray.bitarray(bf_len)
    q_gram_bf1.setall(0)

    for pos in identified_q_gram_pos_dict[freq_q_gram1]:
      q_gram_bf1[pos] = 1

    for freq_q_gram2 in freq_q_gram_list[i+1:]:

      q_gram_bf2 = bitarray.bitarray(bf_len)  # Generate BF for second q-gram
      q_gram_bf2.setall(0)

      for pos in identified_q_gram_pos_dict[freq_q_gram2]:
        q_gram_bf2[pos] = 1

      # Generate the bit-wise OR of both BFs for the q-gram pair
      #
      q_gram_pair_bf = q_gram_bf1 | q_gram_bf2

      # Find all BFs (records) in the encoded data set that potentially encode
      # this q-gram pair
      #
      bf_pair_rec_id_set = set()

      for (enc_rec_id, rec_bf) in encode_bf_dict.iteritems():

        # All 1-bits in the filter must also be set in a record BF
        #
        if (rec_bf & q_gram_pair_bf == q_gram_pair_bf):
          bf_pair_rec_id_set.add(enc_rec_id)

      # Find all records in the plain-text data set that contain both q-grams
      #
      if (len(bf_pair_rec_id_set) > 0):  # Only if there are potential BFs
        q_gram_pair_rec_id_set = set()

        for (plain_rec_id, rec_q_gram_set) in plain_q_gram_dict.iteritems():

          if ((freq_q_gram1 in rec_q_gram_set) and \
              (freq_q_gram2 in rec_q_gram_set)):
            q_gram_pair_rec_id_set.add(plain_rec_id)

      # If both sets are not empty then add them to the dictionaries of q-gram
      # tuples
      #
      if (len(bf_pair_rec_id_set) > 0) and (len(q_gram_pair_rec_id_set) > 0):
        q_gram_pair = (freq_q_gram1, freq_q_gram2)
        q_gram_pair_bf_rec_id_dict[q_gram_pair] =     bf_pair_rec_id_set
        q_gram_pair_q_gram_rec_id_dict[q_gram_pair] = q_gram_pair_rec_id_set

        # Also keep the BF filter for this q-gram pair
        #
        q_gram_pair_bf_dict[q_gram_pair] = q_gram_pair_bf

      else:
        num_q_gram_pair_not_occurring += 1

      auxiliary.check_memory_use(MAX_MEMORY_USE)

  # Calculate statistics of the number of record identifiers per BF and q-gram
  # pair
  #
  num_bf_pair_rec_id_list =     []
  num_q_gram_pair_rec_id_list = []

  for bf_pair_rec_id_set in q_gram_pair_bf_rec_id_dict.itervalues():
    num_bf_pair_rec_id_list.append(len(bf_pair_rec_id_set))
  for q_gram_pair_rec_id_set in q_gram_pair_q_gram_rec_id_dict.itervalues():
    num_q_gram_pair_rec_id_list.append(len(q_gram_pair_rec_id_set))

  print '  %d q-gram pairs had encoded BFs and plain-text attribute ' % \
        (len(q_gram_pair_bf_rec_id_dict)) + 'values'

  if (len(num_bf_pair_rec_id_list) > 0):
    print '    Minimum, average and maximum number of records per frequent' + \
          ' encoded q-gram pair BF: %d / %.2f / %d' % \
          (min(num_bf_pair_rec_id_list), numpy.mean(num_bf_pair_rec_id_list),
           max(num_bf_pair_rec_id_list))
  if (len(num_q_gram_pair_rec_id_list) > 0):
    print '    Minimum, average and maximum number of records per frequent' + \
          ' plain-text q-gram pair: %d / %.2f / %d' % \
          (min(num_q_gram_pair_rec_id_list),
           numpy.mean(num_q_gram_pair_rec_id_list),
           max(num_q_gram_pair_rec_id_list))
  print '    %d pairs of q-grams had no BFs or plain-text attribute values' % \
        (num_q_gram_pair_not_occurring)

  assert len(q_gram_pair_bf_rec_id_dict) == len(q_gram_pair_q_gram_rec_id_dict)

  # Step 2: Now run Apriori for sets of size 3 and more
  #
  curr_len_m1 = 1
  curr_len =    2
  curr_len_p1 = 3

  prev_q_gram_tuple_bf_rec_id_dict =     q_gram_pair_bf_rec_id_dict
  prev_q_gram_tuple_q_gram_rec_id_dict = q_gram_pair_q_gram_rec_id_dict
  prev_q_gram_tuple_bf_dict =            q_gram_pair_bf_dict

  while (len(prev_q_gram_tuple_bf_rec_id_dict) > 1):
    prev_q_gram_tuple_list = sorted(prev_q_gram_tuple_bf_rec_id_dict.keys())

    loop_start_time = time.time()

    # Generate candidates of current length plus 1
    #
    cand_q_gram_tuple_bf_rec_id_dict =     {}
    cand_q_gram_tuple_q_gram_rec_id_dict = {}
    cand_q_gram_tuple_bf_dict =            {}

    for (i, q_gram_tuple1) in enumerate(prev_q_gram_tuple_list[:-1]):

      q_gram_tuple1_m1 =   q_gram_tuple1[:curr_len_m1]
      q_gram_tuple1_last = q_gram_tuple1[-1]

      # Get the BF and q-gram record identifier sets of the first previous
      # q-gram tuple
      #
      prev_tuple_bf_rec_id_set1 = \
                   prev_q_gram_tuple_bf_rec_id_dict[q_gram_tuple1]
      prev_tuple_q_gram_rec_id_set1 = \
                   prev_q_gram_tuple_q_gram_rec_id_dict[q_gram_tuple1]
      prev_tuple_bf_set1 = \
                   prev_q_gram_tuple_bf_dict[q_gram_tuple1]

      for q_gram_tuple2 in prev_q_gram_tuple_list[i+1:]:

        # Check if the two tuples have the same beginning
        #
        if (q_gram_tuple1_m1 == q_gram_tuple2[:curr_len_m1]):
          assert q_gram_tuple1_last < q_gram_tuple2[-1], \
                 (q_gram_tuple1, q_gram_tuple2)
          cand_q_gram_tuple = q_gram_tuple1 + (q_gram_tuple2[-1],)

          # Check all sub-tuples are in previous frequent tuple set
          #
          all_sub_tuple_freq = True
          for j in range(curr_len_p1):
            check_tuple = tuple(cand_q_gram_tuple[:j] + \
                                cand_q_gram_tuple[j+1:])
            if (check_tuple not in prev_q_gram_tuple_bf_rec_id_dict):
              all_sub_tuple_freq = False
              break

          # If all sub-sets are in previous dictionaries, check if there are
          # BFs and plain-text attribute values that contain the candidate
          # q-gram tuple
          #
          if (all_sub_tuple_freq == True):

            # Generate the BF filter for the new q-gram tuple (union of 1-bits)
            #
            cand_tuple_bf = prev_tuple_bf_set1 | \
                                  prev_q_gram_tuple_bf_dict[q_gram_tuple2]

            # Get all the records in the encoded data set that potentially can
            # encode the new q-gram tuple
            #
            cand_bf_rec_id_set = prev_tuple_bf_rec_id_set1 & \
                         prev_q_gram_tuple_bf_rec_id_dict[q_gram_tuple2]

            # If there are potential BFs check if there are also attribute
            # values
            #
            if (len(cand_bf_rec_id_set) > 0):

              # The record identifiers from the plain-text data set that
              # potentially can contain all q-grams in the candidate tuple
              #
              cand_q_gram_tuple_rec_id_set = prev_tuple_q_gram_rec_id_set1 & \
                        prev_q_gram_tuple_q_gram_rec_id_dict[q_gram_tuple2]

              cand_q_gram_rec_id_set = set()

              # Get identifiers of the records that contain all q-grams in the
              # candidate q-gram tuple
              #
              for plain_rec_id in cand_q_gram_tuple_rec_id_set:
                cand_q_gram_rec_id_set.add(plain_rec_id)

              # If there are also potential attribute values then add them to
              # the dictionaries of q-gram tuples of the current length
              #
              if (len(cand_q_gram_rec_id_set) > 0):

                assert cand_q_gram_tuple not in cand_q_gram_tuple_bf_rec_id_dict

                cand_q_gram_tuple_bf_rec_id_dict[cand_q_gram_tuple] = \
                                 cand_bf_rec_id_set
                cand_q_gram_tuple_q_gram_rec_id_dict[cand_q_gram_tuple] = \
                                 cand_q_gram_rec_id_set
                cand_q_gram_tuple_bf_dict[cand_q_gram_tuple] = cand_tuple_bf

    if (len(cand_q_gram_tuple_bf_dict) == 0):
      break  # No more candidates, end Apriori process


# PC 20171010: only remove if less attr values and less BFs - but this will
# likely mean all tuples will be kept! *********************


    # Remove all shorter q-gram tuples that are true sub-sets of one of the
    # new longer tuples
    #
    num_del_shorter_tuple = 0

    for long_q_gram_tuple in cand_q_gram_tuple_bf_dict.iterkeys():
      long_q_gram_tuple_set = set(long_q_gram_tuple)

      for short_q_gram_tuple in q_gram_tuple_rec_id_dict.keys():
        if (len(short_q_gram_tuple) == curr_len):
          if (set(short_q_gram_tuple).issubset(long_q_gram_tuple_set)):
            del q_gram_tuple_rec_id_dict[short_q_gram_tuple]
            num_del_shorter_tuple += 1

    print '    Removed %d q-gram tuples of length %d that are true sub-sets' \
          % (num_del_shorter_tuple, curr_len)+' of q-gram tuples of length', \
          curr_len_p1
    print

    # ************************************************************************
    # PC TODO 20170924: the above code is quadratic - a better way? During the
    # generation of longer tuples?
    # ************************************************************************

    # Add to the final dictionary to be returned
    #
    for q_gram_tuple in cand_q_gram_tuple_bf_rec_id_dict:
      bf_tuple_rec_id_set = cand_q_gram_tuple_bf_rec_id_dict[q_gram_tuple]
      q_gram_rec_id_set =   cand_q_gram_tuple_q_gram_rec_id_dict[q_gram_tuple]

      q_gram_tuple_rec_id_dict[q_gram_tuple] = (bf_tuple_rec_id_set,
                                                q_gram_rec_id_set)

    num_bf_tuple_rec_id_list =     []  # Calculate statistics of sets
    num_q_gram_tuple_rec_id_list = []

    for bf_tuple_rec_id_set in cand_q_gram_tuple_bf_rec_id_dict.itervalues():
      num_bf_tuple_rec_id_list.append(len(bf_tuple_rec_id_set))
    for q_gram_rec_id_set in cand_q_gram_tuple_q_gram_rec_id_dict.itervalues():
      num_q_gram_tuple_rec_id_list.append(len(q_gram_rec_id_set))

    print '  %d q-gram tuples of size %d had encoded BFs and plain-text ' % \
          (len(cand_q_gram_tuple_bf_rec_id_dict), curr_len_p1) + \
          'attribute values'
    if (len(num_bf_tuple_rec_id_list) > 0):
      print '    Minimum, average and maximum number of records per frequent' \
            + ' encoded q-gram tuple BF: %d / %.2f / %d' % \
            (min(num_bf_tuple_rec_id_list),
             numpy.mean(num_bf_tuple_rec_id_list),
             max(num_bf_tuple_rec_id_list))
    if (len(num_q_gram_tuple_rec_id_list) > 0):
      print '    Minimum, average and maximum number of records per frequent' \
            + ' plain-text q-gram tuple: %d / %.2f / %d' % \
            (min(num_q_gram_tuple_rec_id_list), 
             numpy.mean(num_q_gram_tuple_rec_id_list),
             max(num_q_gram_tuple_rec_id_list))
    print

    auxiliary.check_memory_use(MAX_MEMORY_USE)

    prev_q_gram_tuple_bf_rec_id_dict =     cand_q_gram_tuple_bf_rec_id_dict
    prev_q_gram_tuple_q_gram_rec_id_dict = cand_q_gram_tuple_q_gram_rec_id_dict
    prev_q_gram_tuple_bf_dict =            cand_q_gram_tuple_bf_dict

    curr_len_m1 += 1
    curr_len +=    1
    curr_len_p1 += 1

  # Now check if we can remove plain-text values (records) because they contain
  # cannot have q-grams
  #
  checked_q_gram_tuple_rec_id_dict = {}

  for q_gram_tuple in q_gram_tuple_rec_id_dict:
    enc_rec_id_set =   q_gram_tuple_rec_id_dict[q_gram_tuple][0]
    plain_rec_id_set = q_gram_tuple_rec_id_dict[q_gram_tuple][1]

    # For each encoded record check each plain-text record
    #
    enc_rec_plain_rec_id_set = set()

    for enc_rec_id in enc_rec_id_set:
      cannot_q_gram_set = bf_cannot_have_q_gram_dict.get(enc_rec_id, set())

      # Check if a plain-text record contains cannot have q-grams
      #
      for plain_rec_id in plain_rec_id_set:
        plain_rec_q_gram_set = plain_q_gram_dict[plain_rec_id]

        no_cannot_q_gram = True
        for q_gram in cannot_q_gram_set:
          if q_gram in plain_rec_q_gram_set:
            no_cannot_q_gram = False
            break

        if (no_cannot_q_gram == True):
          enc_rec_plain_rec_id_set.add(plain_rec_id)

      if (len(enc_rec_plain_rec_id_set) > 0):  # There are plain-text records

        # Generate a new unique key for this encoded BF
        #
        checked_q_gram_tuple = q_gram_tuple + (enc_rec_id,)

        checked_q_gram_tuple_rec_id_dict[checked_q_gram_tuple] = \
                      (set([enc_rec_id]), enc_rec_plain_rec_id_set)

  print '  Identified %d q-gram tuples that match both encoded BFs and ' % \
        (len(checked_q_gram_tuple_rec_id_dict)) + 'plain-text attribute values'
  print '    Overall matching of BFs and attribute values took %.1f sec' % \
        (time.time()-start_time)
  print '   ', auxiliary.get_memory_usage()
  print

  return checked_q_gram_tuple_rec_id_dict

# -----------------------------------------------------------------------------

def get_matching_bf_sets(identified_q_gram_pos_dict, encode_bf_dict,
                         plain_attr_val_rec_id_dict, bf_must_have_q_gram_dict,
                         bf_cannot_have_q_gram_dict, bf_len,
                         min_q_gram_tuple_size=3):
  """Based on the given identified bit position tuples of frequent q-grams, the
     given BF dictionary (assumed to come from the encoded data set), and the
     given q-gram dictionary (assumed to contain the q-grams from records in
     the plain-text data set), as well as the must have and cannot have q-gram
     sets per BF, for each encoded BF first find all the q-grams that can be
     encoded in this BF (based on the BF's 1-bit pattern), and then find for
     each unique q-gram tuple and its possible BFs the possible matching
     attribute values.

     The function returns a dictionary where frequent q-gram tuples are keys,
     and values are two sets of record identifiers, one of encoded BFs that
     have matching 1-bits in all relevant positions for the q-grams in the
     key, and the second with record identifiers from the plain-text data set
     that contain all the q-grams in the key.
  """

  start_time = time.time()

  # The dictionary to be returned with q-gram tuples as keys and two sets of
  # record identifiers (one corresponding to encoded BFs, the other to
  # plain-text values) for each such q-gram tuples.
  #
  q_gram_tuple_rec_id_dict = {}

  # The list of frequent q-grams we have
  #
  freq_q_gram_set = set(identified_q_gram_pos_dict.keys())

  print 'Find q-gram tuples that have corresponding BFs and attribute ' + \
        'values:'
  print '  %d frequent q-grams:' % (len(freq_q_gram_set)), \
                                    sorted(freq_q_gram_set)
  print '  Only consider q-gram tuples of size at least %d' % \
        (min_q_gram_tuple_size) + ' (unless they correspond to only one BF)'
  print

  # Step 1: For each BF, find all frequent q-grams that possibly could be
  #         encoded in this BF, not just the must have ones
  #
  q_gram_tuple_enc_rec_set_dict = {}  # Keys are q-gram tuples, values record
                                      # ID sets from the encoded database

  for (enc_rec_id, rec_bf) in encode_bf_dict.iteritems():
    must_have_q_gram_set =   bf_must_have_q_gram_dict.get(enc_rec_id, set())
    cannot_have_q_gram_set = bf_cannot_have_q_gram_dict.get(enc_rec_id, set())

    bf_q_gram_set = must_have_q_gram_set.copy()  # Start with the ones we know

    # Get the set of other q-grams that might be encoded (from all identified
    # ones)
    #
    check_q_gram_set = freq_q_gram_set - must_have_q_gram_set - \
                       cannot_have_q_gram_set

    if (len(check_q_gram_set) > 0):

      for q_gram in check_q_gram_set:
        all_q_gram_pos_1 = True

        for pos in identified_q_gram_pos_dict[q_gram]:
          if (rec_bf[pos] == 0):
            all_q_gram_pos_1 = False

        if (all_q_gram_pos_1 == True):
          print '    Added "%s" as possible q-gram to encoded BF %s' % \
                (q_gram,enc_rec_id)
          bf_q_gram_set.add(q_gram)

    if (len(bf_q_gram_set) > 0):
      bf_q_gram_tuple = tuple(bf_q_gram_set)

      bf_q_gram_tuple_set_id_set = \
                    q_gram_tuple_enc_rec_set_dict.get(bf_q_gram_tuple, set())
      bf_q_gram_tuple_set_id_set.add(enc_rec_id)
      q_gram_tuple_enc_rec_set_dict[bf_q_gram_tuple] = \
                                                   bf_q_gram_tuple_set_id_set

  # Calculate statistics of the number of encoded BFs / record identifiers per
  # q-gram tuple
  #
  num_enc_rec_id_list = []
  for q_gram_tuple_rec_id_set in q_gram_tuple_enc_rec_set_dict.itervalues():
    num_enc_rec_id_list.append(len(q_gram_tuple_rec_id_set))

  print '  Identified %d q-gram tuples from encoded BF' % \
        (len(q_gram_tuple_enc_rec_set_dict))
  if (len(num_enc_rec_id_list) > 0):
    print '    Minimum, average, maximum numbers of BFs with these q-grams: ' \
          + '%.2f min / %.2f avr / %.2f max' % \
          (min(num_enc_rec_id_list), numpy.mean(num_enc_rec_id_list),
           max(num_enc_rec_id_list))
  print

  # Remove q-grams tuples that are not long enough
  #
  num_short_tuples_del = 0
 
  num_many_del = 0

  for q_gram_tuple in q_gram_tuple_enc_rec_set_dict.keys():
    if (len(q_gram_tuple) < min_q_gram_tuple_size):

      # If the tuple has more than one encoded BF remove it
      #
      if (len(q_gram_tuple_enc_rec_set_dict[q_gram_tuple]) > 1):
        del q_gram_tuple_enc_rec_set_dict[q_gram_tuple]
        num_short_tuples_del += 1

# -------- PC 20171010: experiments showed too many re-identifications lost
#    # Experimental: also remove if too many encoded record identifiers
#    #
#    if (q_gram_tuple in q_gram_tuple_enc_rec_set_dict):
#      if (len(q_gram_tuple_enc_rec_set_dict[q_gram_tuple]) > max_num_many):
#        del q_gram_tuple_enc_rec_set_dict[q_gram_tuple]
#        num_many_del += 1
#
#  print '  Removed %d q-gram tuples that had more than %d record identifiers' \
#        % (num_many_del, max_num_many)
# ------------

  print '  Removed %d q-gram tuples with less than %d q-grams' % \
        (num_short_tuples_del, min_q_gram_tuple_size)

  num_enc_rec_id_list = []
  for q_gram_tuple_rec_id_set in q_gram_tuple_enc_rec_set_dict.itervalues():
    num_enc_rec_id_list.append(len(q_gram_tuple_rec_id_set))

  if (len(num_enc_rec_id_list) > 0):
    print '    Minimum, average, maximum numbers of BFs with these q-grams: ' \
          + '%.2f min / %.2f avr / %.2f max' % \
          (min(num_enc_rec_id_list), numpy.mean(num_enc_rec_id_list),
           max(num_enc_rec_id_list))
    print

  print '  Number of unique plain-text attribute values: %d' % \
        (len(plain_attr_val_rec_id_dict))

  # Step 2: For each found q-gram tuple which has encoded BFs, get the
  #         plain-text attribute values with these q-grams, and then the
  #         corresponding record identifiers from the plain-text database
  #
  q_gram_tuple_plain_rec_set_dict = {}  # Keys are q-gram tuples, values record
                                        # ID sets from the plain-text database
  q_gram_tuple_plain_attr_val_dict = {}  # Keys are q-gram tuples, values are
                                         # plain-text attribute values

  num_no_matching_q_grams =      0
  num_same_length_q_gram_tuple = 0

  q_gram_tuple_len_list = []
  for q_gram_tuple in q_gram_tuple_enc_rec_set_dict.iterkeys():
    q_gram_tuple_len_list.append((q_gram_tuple, len(q_gram_tuple)))

  # Loop over all attribute values from the plain-text database and their
  # record identifiers, and find the q-gram tuples that have all q-grams in an
  # attribute value
  #
  for (attr_val, plain_rec_id_set) in plain_attr_val_rec_id_dict.iteritems():

    # Keep all matching q-gram tuples and their length (number of q-grams)
    #
    attr_val_q_gram_tuple_len_list = []

    for (q_gram_tuple, tuple_len) in q_gram_tuple_len_list:

      all_q_grams_in_val = True

      for q_gram in q_gram_tuple:
        if (q_gram not in attr_val):
          all_q_grams_in_val = False

      if (all_q_grams_in_val == True):
        attr_val_q_gram_tuple_len_list.append((tuple_len, q_gram_tuple, \
                                              plain_rec_id_set))

    # Now get the longest tuple with the largest number of q-grams
    #
    if (len(attr_val_q_gram_tuple_len_list) == 1):  # Only one matching tuple

      q_gram_tuple = attr_val_q_gram_tuple_len_list[0][1]

      # There could be several attribute values that match this q-gram tuple
      #
      q_gram_tuple_attr_val_set = \
                   q_gram_tuple_plain_attr_val_dict.get(q_gram_tuple, set())
      q_gram_tuple_attr_val_set.add(attr_val)
      q_gram_tuple_plain_attr_val_dict[q_gram_tuple] = \
                                               q_gram_tuple_attr_val_set
      q_gram_tuple_rec_id_set = \
                   q_gram_tuple_plain_rec_set_dict.get(q_gram_tuple, set())
      q_gram_tuple_rec_id_set.update(attr_val_q_gram_tuple_len_list[0][2])
      q_gram_tuple_plain_rec_set_dict[q_gram_tuple] = q_gram_tuple_rec_id_set

    elif (len(attr_val_q_gram_tuple_len_list) > 1):
      attr_val_q_gram_tuple_len_list.sort(reverse=True)  # Longest tuple first

      # Only use the longest q-gram tuple (if there is one longest one)
      #
      if (attr_val_q_gram_tuple_len_list[0][0] > \
          attr_val_q_gram_tuple_len_list[1][0]):
        q_gram_tuple = attr_val_q_gram_tuple_len_list[0][1]

        q_gram_tuple_attr_val_set = \
                     q_gram_tuple_plain_attr_val_dict.get(q_gram_tuple, set())
        q_gram_tuple_attr_val_set.add(attr_val)
        q_gram_tuple_plain_attr_val_dict[q_gram_tuple] = \
                                               q_gram_tuple_attr_val_set
        q_gram_tuple_rec_id_set = \
                   q_gram_tuple_plain_rec_set_dict.get(q_gram_tuple, set())
        q_gram_tuple_rec_id_set.update(attr_val_q_gram_tuple_len_list[0][2])
        q_gram_tuple_plain_rec_set_dict[q_gram_tuple] = q_gram_tuple_rec_id_set

      else:
        num_same_length_q_gram_tuple += 1


# TODO - what here? ************************


    else:
      num_no_matching_q_grams += 1

# PC TODO 20171011: how to handle several tuples of same length? **********

#      i = 1
#      while (i < len(q_gram_tuple_len_list)) and \
#            (q_gram_tuple_len_list[i-1][0] > q_gram_tuple_len_list[i][0]):
#        attr_val_plain_rec_id_set.update(q_gram_tuple_len_list[i][2])
#        i += 1

  # Calculate statistics of the number of plain-text record identifiers per
  # q-gram tuple
  #
  num_plain_rec_id_list = []
  for q_gram_tuple_rec_id_set in q_gram_tuple_plain_rec_set_dict.itervalues():
    num_plain_rec_id_list.append(len(q_gram_tuple_rec_id_set))

  print '  Identified %d q-gram tuples from plain-text database matching' % \
        (len(q_gram_tuple_plain_rec_set_dict)) + \
        ' q-gram tuples from BF database'
  if (len(num_plain_rec_id_list) > 0):
    print '    Minimum, average, maximum numbers of BFs with these q-grams:' \
          +    '%.2f min / %.2f avr / %.2f max' % \
          (min(num_plain_rec_id_list), numpy.mean(num_plain_rec_id_list),
           max(num_plain_rec_id_list))
  print '    Number of q-gram tuples with 1 plain-text attribute: ' + \
        '%d' % (num_plain_rec_id_list.count(1))
  print
  print '    Number of attribute values with no matching q-gram tuples: ' + \
        '%d' % (num_no_matching_q_grams)
  print '    Number of attribute values with multiple longest q-gram ' + \
        'tuples: %d' % (num_same_length_q_gram_tuple)
  print

  # Generate final dictionary to be returned
  #
  for (q_gram_tuple, enc_q_gram_tuple_rec_id_set) in \
                             q_gram_tuple_enc_rec_set_dict.iteritems():
    if (q_gram_tuple in q_gram_tuple_plain_rec_set_dict):
      plain_q_gram_tuple_rec_id_set = \
                             q_gram_tuple_plain_rec_set_dict[q_gram_tuple]
      q_gram_tuple_rec_id_dict[q_gram_tuple] = \
          (enc_q_gram_tuple_rec_id_set, plain_q_gram_tuple_rec_id_set)

  print '    Overall matching of BFs and attribute values took %.1f sec' % \
        (time.time()-start_time)
  print '   ', auxiliary.get_memory_usage()
  print

  return q_gram_tuple_rec_id_dict

# -----------------------------------------------------------------------------

def calc_reident_accuracy(bf_q_gram_rec_id_dict, encode_rec_val_dict,
                          plain_rec_val_dict, max_num_many=10):
  """Calculate the accuracy of re-identification for the given dictionary that
     contains q-grams as keys and pairs of record identifier sets (one from
     the encoded data set the other from the plain-text data set), where the
     former are BFs that are believed to encode these q-grams while the latter
     are records that contain these q-grams.

     Calculate and return the number of:
     - BFs with no guesses
     - BFs with more than 'max_num_many' guesses
     - BFs with 1-to-1 guesses
     - BFs with correct 1-to-1 guesses
     - BFs with partially matching 1-to-1 guesses
     - BFs with 1-to-many guesses
     - BFs with 1-to-many correct guesses
     - BFs with partially matching 1-to-many guesses

     - Accuracy of 1-to-1 partial matching values based on common tokens
     - Accuracy of 1-to-many partial matching values based on common tokens

     Also returns a dictionary with BFs as keys and correctly re-identified
     attribute values as values.
  """

  print 'Re-identify encoded attribute values based q-gram tuples and ' + \
        'corresponding encoded and plain-text records:'

  start_time = time.time()

  num_no_guess =       0
  num_too_many_guess = 0
  num_1_1_guess =      0
  num_corr_1_1_guess = 0
  num_part_1_1_guess = 0
  num_1_m_guess =      0
  num_corr_1_m_guess = 0
  num_part_1_m_guess = 0

  acc_part_1_1_guess = 0.0  # Average accuracy of partial matching values based
  acc_part_1_m_guess = 0.0  # on common tokens

  # BFs with correctly re-identified attribute values
  #
  corr_reid_attr_val_dict = {}

  # First get for each encoded BF all the plain-text record identifiers
  #
  encode_plain_rec_id_dict = {}

  for (q_gram_tuple, attr_val_rec_id_sets_pair) in \
                    bf_q_gram_rec_id_dict.iteritems():
    bf_rec_id_set =     attr_val_rec_id_sets_pair[0]
    q_gram_rec_id_set = attr_val_rec_id_sets_pair[1]

    for bf_rec_id in bf_rec_id_set:
      bf_plain_rec_set = encode_plain_rec_id_dict.get(bf_rec_id, set())
      bf_plain_rec_set.update(q_gram_rec_id_set)
      encode_plain_rec_id_dict[bf_rec_id] = bf_plain_rec_set

  # Now loop over these encoded BFs and their plain-text record identifier sets
  #
  for (bf_rec_id, bf_plain_rec_set) in encode_plain_rec_id_dict.iteritems():

    # First get all the plain-text attribute values from the corresponding
    # records
    #
    q_gram_plain_attr_val_set = set()

    for rec_id in bf_plain_rec_set:
      q_gram_plain_attr_val_set.add(plain_rec_val_dict[rec_id])

    num_plain_val = len(q_gram_plain_attr_val_set)

    # Now check for the encoded BF record if the plain text values match
    #
    true_encoded_attr_val = encode_rec_val_dict[bf_rec_id]

    if (num_plain_val == 1):
      num_1_1_guess += 1
    elif (num_plain_val > max_num_many):
      num_too_many_guess += 1
    else:
      num_1_m_guess += 1

    if (num_plain_val >= 1) and (num_plain_val <= max_num_many):

      if (true_encoded_attr_val in q_gram_plain_attr_val_set):

        # True attribute value is re-identified
        #
        corr_reid_attr_val_dict[rec_id] = q_gram_plain_attr_val_set

        if (num_plain_val == 1):
          num_corr_1_1_guess += 1
        else:
          num_corr_1_m_guess += 1

      else:  # If no exact match, check if some words / tokens are in common
        true_encoded_attr_val_set = set(true_encoded_attr_val.split())

        # Get maximum number of tokens shared with an encoded attribute value
        #
        max_num_common_token = 0

        for plain_text_attr_val in q_gram_plain_attr_val_set:
          plain_text_attr_val_set = set(plain_text_attr_val.split())

          num_common_token = \
                     len(true_encoded_attr_val_set & plain_text_attr_val_set)
          max_num_common_token = max(max_num_common_token, num_common_token)

        if (max_num_common_token > 0):  # Add partial accuracy
          num_token_acc = float(max_num_common_token) / \
                               len(true_encoded_attr_val_set)

          if (num_plain_val == 1):
            num_part_1_1_guess += 1
            acc_part_1_1_guess += num_token_acc
          else:
            num_part_1_m_guess += 1
            acc_part_1_m_guess += num_token_acc

  total_time = time.time() - start_time

  if (num_part_1_m_guess > 0):
    acc_part_1_m_guess = float(acc_part_1_m_guess) / num_part_1_m_guess
  else:
    acc_part_1_m_guess = 0.0
  if (num_part_1_1_guess > 0):
    acc_part_1_1_guess = float(acc_part_1_1_guess) / num_part_1_1_guess
  else:
    acc_part_1_1_guess = 0.0

  print '  Total time required to re-identify from %d q-gram tuples: ' % \
        (len(bf_q_gram_rec_id_dict)) + '%.1f sec' % (total_time)
  print
  print '  Num no guesses:                          %d' % (num_no_guess)
  print '  Num > %d guesses:                        %d' % \
        (max_num_many, num_too_many_guess)
  print '  Num 2 to %d guesses:                     %d' % \
        (max_num_many, num_1_m_guess)
  print '    Num correct 2 to %d guesses:           %d' % \
        (max_num_many, num_corr_1_m_guess)
  if (num_part_1_m_guess > 0):
    print '    Num partially correct 2 to %d guesses: %d' % \
          (max_num_many, num_part_1_m_guess) + \
          ' (average accuracy of common tokens: %.2f)' % (acc_part_1_m_guess)
  else:
    print '    No partially correct 2 to %d guesses' % (max_num_many)
  print '  Num 1-1 guesses:                         %d' % \
        (num_1_1_guess)
  print '    Num correct 1-1 guesses:               %d' % \
        (num_corr_1_1_guess)
  if (num_part_1_1_guess > 0):
    print '    Num partially correct 1-1 guesses:     %d' % \
          (num_part_1_1_guess) + \
          ' (average accuracy of common tokens: %.2f)' % (acc_part_1_1_guess)
  else:
    print '    No partially correct 1-1 guesses'
  print

  return num_no_guess, num_too_many_guess, num_1_1_guess, num_corr_1_1_guess, \
         num_part_1_1_guess, num_1_m_guess, num_corr_1_m_guess, \
         num_part_1_m_guess, acc_part_1_1_guess, acc_part_1_m_guess, \
         corr_reid_attr_val_dict

# =============================================================================
# Main program

q =                      int(sys.argv[1])
hash_type =              sys.argv[2].lower()
num_hash_funct =         sys.argv[3]
bf_len =                 int(sys.argv[4])
bf_harden =              sys.argv[5].lower()
stop_iter_perc =         float(sys.argv[6])
min_part_size =          int(sys.argv[7])
#
encode_data_set_name =    sys.argv[8]
encode_rec_id_col =       int(sys.argv[9])
encode_col_sep_char =     sys.argv[10]
encode_header_line_flag = eval(sys.argv[11])
encode_attr_list =        eval(sys.argv[12])
#
plain_data_set_name =    sys.argv[13]
plain_rec_id_col =       int(sys.argv[14])
plain_col_sep_char =     sys.argv[15]
plain_header_line_flag = eval(sys.argv[16])
plain_attr_list =        eval(sys.argv[17])
max_num_many =           int(sys.argv[18])
print sys.argv[18]
re_id_method =           sys.argv[19]

assert q >= 1, q
assert hash_type in ['dh', 'rh'], hash_type
if num_hash_funct.isdigit():
  num_hash_funct = int(num_hash_funct)
  assert num_hash_funct >= 1, num_hash_funct
else:
  assert num_hash_funct == 'opt', num_hash_funct
assert bf_len > 1, bf_len
assert bf_harden in ['none', 'balance', 'fold'], bf_harden
assert stop_iter_perc > 0.0 and stop_iter_perc < 100.0, stop_iter_perc
assert min_part_size > 1, min_part_size

assert encode_rec_id_col >= 0, encode_rec_id_col
assert encode_header_line_flag in [True, False], encode_header_line_flag
assert isinstance(encode_attr_list, list), encode_attr_list
#
assert plain_rec_id_col >= 0, plain_rec_id_col
assert plain_header_line_flag in [True, False], plain_header_line_flag
assert isinstance(plain_attr_list, list), plain_attr_list

assert max_num_many > 1, max_num_many
assert re_id_method in ['all', 'set_inter', 'apriori', 'q_gram_tuple', \
                        'bf_q_gram_tuple', 'bf_tuple', 'none']

if (bf_harden == 'fold'):
  if (bf_len%2 != 0):
    raise Exception, 'BF hardening approach "fold" needs an even BF length'

if (len(encode_col_sep_char) > 1):
  if (encode_col_sep_char == 'tab'):
    encode_col_sep_char = '\t'
  elif (encode_col_sep_char[0] == '"') and (encode_col_sep_char[-1] == '"') \
       and (len(encode_col_sep_char) == 3):
    encode_col_sep_char = encode_col_sep_char[1]
  else:
    print 'Illegal encode data set column separator format:', \
          encode_col_sep_char

if (len(plain_col_sep_char) > 1):
  if (plain_col_sep_char == 'tab'):
    plain_col_sep_char = '\t'
  elif (plain_col_sep_char[0] == '"') and \
     (plain_col_sep_char[-1] == '"') and \
     (len(plain_col_sep_char) == 3):
    plain_col_sep_char = plain_col_sep_char[1]
  else:
    print 'Illegal plain text data set column separator format:', \
          plain_col_sep_char

# Check if same data sets and same attributes were given
#
if ((encode_data_set_name == plain_data_set_name) and \
    (encode_attr_list == plain_attr_list)):
  same_data_attr_flag = True
else:
  same_data_attr_flag = False

# Get base names of data sets (remove directory names) for summary output
#
encode_base_data_set_name = encode_data_set_name.split('/')[-1]
encode_base_data_set_name = encode_base_data_set_name.replace('.csv', '')
encode_base_data_set_name = encode_base_data_set_name.replace('.gz', '')
assert ',' not in encode_base_data_set_name

plain_base_data_set_name = plain_data_set_name.split('/')[-1]
plain_base_data_set_name = plain_base_data_set_name.replace('.csv', '')
plain_base_data_set_name = plain_base_data_set_name.replace('.gz', '')
assert ',' not in plain_base_data_set_name

res_file_name = 'bf-column-attack-results-%s-%s-%s.csv' % \
                (encode_base_data_set_name, plain_base_data_set_name, \
                 today_str)
print
print 'Write results into file:', res_file_name
print
print '-'*80
print

# -----------------------------------------------------------------------------
# Step 1: Load the data sets and extract q-grams for selected attributes
#
start_time = time.time()

encode_q_gram_res_tuple = load_data_set_extract_q_grams(encode_data_set_name,
                                                      encode_rec_id_col,
                                                      encode_attr_list,
                                                      encode_col_sep_char,
                                                      encode_header_line_flag,
                                                      q, DO_PADDING)
encode_rec_val_dict =         encode_q_gram_res_tuple[0]
encode_q_gram_dict =          encode_q_gram_res_tuple[1]
encode_unique_q_gram_set =    encode_q_gram_res_tuple[2]
encode_avrg_num_q_gram =      encode_q_gram_res_tuple[3]
encode_stdd_num_q_gram =      encode_q_gram_res_tuple[4]
encode_q_gram_freq_dict =     encode_q_gram_res_tuple[5]
encode_q_gram_attr_val_dict = encode_q_gram_res_tuple[6]
encode_attr_val_rec_id_dict = encode_q_gram_res_tuple[7]
encode_attr_name_list =       encode_q_gram_res_tuple[8]

encode_load_time = time.time() - start_time

if (same_data_attr_flag == False):
  start_time = time.time()

  plain_q_gram_res_tuple = \
                  load_data_set_extract_q_grams(plain_data_set_name,
                                                plain_rec_id_col,
                                                plain_attr_list,
                                                plain_col_sep_char,
                                                plain_header_line_flag,
                                                q, DO_PADDING)
  plain_rec_val_dict =         plain_q_gram_res_tuple[0]
  plain_q_gram_dict =          plain_q_gram_res_tuple[1]
  plain_unique_q_gram_set =    plain_q_gram_res_tuple[2]
  plain_avrg_num_q_gram =      plain_q_gram_res_tuple[3]
  plain_stdd_num_q_gram =      plain_q_gram_res_tuple[4]
  plain_q_gram_freq_dict =     plain_q_gram_res_tuple[5]
  plain_q_gram_attr_val_dict = plain_q_gram_res_tuple[6]
  plain_attr_val_rec_id_dict = plain_q_gram_res_tuple[7]
  plain_attr_name_list =       plain_q_gram_res_tuple[8]

  plain_load_time = time.time() - start_time

  if (encode_attr_name_list != plain_attr_name_list):
    print '*** Warning: Different attributes used to encode BF and plain text:'
    print '***   BF encode: ', encode_attr_name_list
    print '***   Plain text:', plain_attr_name_list

else:  # Set to same as encode
  plain_rec_val_dict =         encode_rec_val_dict
  plain_q_gram_dict =          encode_q_gram_dict
  plain_unique_q_gram_set =    encode_unique_q_gram_set
  plain_avrg_num_q_gram =      encode_avrg_num_q_gram
  plain_stdd_num_q_gram =      encode_stdd_num_q_gram
  plain_q_gram_freq_dict =     encode_q_gram_freq_dict
  plain_q_gram_attr_val_dict = encode_q_gram_attr_val_dict
  plain_attr_val_rec_id_dict = encode_attr_val_rec_id_dict
  plain_attr_name_list =       encode_attr_name_list

plain_num_rec = len(plain_rec_val_dict)

# Find how many attribute values are in common (exactly) across the two data
# sets (as this gives an upper bound on re-identification accuracy
#
encode_attr_val_set = set(encode_attr_val_rec_id_dict.keys())
plain_attr_val_set =  set(plain_attr_val_rec_id_dict.keys())

common_attr_val_set = encode_attr_val_set & plain_attr_val_set

print 'Number of unique attribute values in data sets and in common:'
print '  %d in the encoded data set'    % (len(encode_attr_val_set))
print '  %d in the plain-text data set' % (len(plain_attr_val_set))
perc_comm = 200.0*float(len(common_attr_val_set)) / \
            (len(encode_attr_val_set) + len(plain_attr_val_set))
print '  %d occur in both data sets (%2.f%%)' % (len(common_attr_val_set),
      perc_comm)
print

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Generate the q-gram support graph (a dictionary of nodes (q-grams) and their
# counts, and a dictionary of edges (q-gram pairs) and their counts
#
plain_q_gram_node_dict, plain_q_gram_edge_dict = \
                  gen_q_gram_supp_graph(plain_unique_q_gram_set,
                                        plain_q_gram_dict)

# -----------------------------------------------------------------------------
# Step 2: Generate Bloom filters for records (for both data sets so we know the
#         true mapping of q-grams to BF bit positions)
#
start_time = time.time()

if (num_hash_funct == 'opt'):

  # Set number of hash functions to have in average 50% of bits set to 1
  # (reference to published paper? Only in Dinusha's submitted papers) 
  # num_hash_funct = int(math.ceil(0.5 * BF_LEN / \
  #                                math.floor(avrg_num_q_gram)))
  #
  num_hash_funct = int(round(numpy.log(2.0) * float(bf_len) /
                                encode_avrg_num_q_gram))

encode_bf_dict, encode_true_q_gram_pos_map_dict = \
               gen_bloom_filter_dict(encode_q_gram_dict, hash_type, bf_len,
                                     num_hash_funct)
encode_num_bf = len(encode_bf_dict)

encode_bf_gen_time = time.time() - start_time

if (same_data_attr_flag == False):
  start_time = time.time()

  plain_bf_dict, plain_true_q_gram_pos_map_dict = \
                 gen_bloom_filter_dict(plain_q_gram_dict, hash_type, bf_len,
                                       num_hash_funct)
  plain_num_bf = len(plain_bf_dict)

  plain_bf_gen_time = time.time() - start_time

else:  # Use same as build
  plain_bf_dict =                  encode_bf_dict
  plain_true_q_gram_pos_map_dict = encode_true_q_gram_pos_map_dict
  plain_num_bf =                   encode_num_bf

if (bf_harden == 'balance'):
  encode_bf_dict, encode_true_q_gram_pos_map_dict = \
      balance_bf_dict(encode_bf_dict, bf_len, encode_true_q_gram_pos_map_dict)
  if (same_data_attr_flag == False):
    plain_bf_dict, plain_true_q_gram_pos_map_dict = \
      balance_bf_dict(plain_bf_dict, bf_len, plain_true_q_gram_pos_map_dict)
  else:  # Use same as build
    plain_bf_dict = encode_bf_dict
  bf_len *= 2

elif (bf_harden == 'fold'):
  encode_bf_dict, encode_true_q_gram_pos_map_dict = \
        fold_bf_dict(encode_bf_dict, bf_len, encode_true_q_gram_pos_map_dict)
  if (same_data_attr_flag == False):
    plain_bf_dict, plain_true_q_gram_pos_map_dict = \
        fold_bf_dict(plain_bf_dict, bf_len, plain_true_q_gram_pos_map_dict)
  else:  # Use same as build
    plain_bf_dict = encode_bf_dict
  bf_len = int(bf_len / 2.0)

# Calculate the probability that two q-grams are hashed into the same column
# (using the birthday paradox) assuming each q-gram is hashed 'num_hash_funct'
# times
#
no_same_col_prob = 1.0
for i in range(1,2*num_hash_funct):
  no_same_col_prob *= float(bf_len-i) / bf_len
print 'Birthday paradox probability that two q-grams are hashed to the ' + \
      'same bit position p = %.5f' % (1-no_same_col_prob)
print

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Convert the encoded BF dictionary into column-wise storage
#
encode_bf_bit_pos_list, encode_rec_id_list = \
                          gen_bf_col_dict(encode_bf_dict, bf_len)

# *************************************************************************
# PC 201709017: The above conversion of row-wise BF into column-wise is
# very slow - how to make it faster?
# ************************************************************************

# Get the frequency distribution of how often each BF row and column occurs
#
row_count_dict, col_count_dict = get_bf_row_col_freq_dist(encode_bf_dict,
                                                      encode_bf_bit_pos_list)
most_freq_bf_pattern_count =         max(row_count_dict.keys())
most_freq_bf_bit_pos_pattern_count = max(col_count_dict.keys())

# Calculate and print the average Hamming weight for pairs and triplets of
# randomly selected bit positions
#
check_hamming_weight_bit_positions(encode_bf_bit_pos_list, NUM_SAMPLE)

# Check if the BF set was hardened using BF balancing
#
encode_bit_pos_pair_set = check_balanced_bf(encode_bf_bit_pos_list)

# -----------------------------------------------------------------------------
# Step 3: Recursively find most frequent q-gram, then BF bit positions that are
#         frequent, assign q-gram to them, split BF set into two and repeat
#         process.

start_time = time.time()

# A dictionary of how q-grams have been assigned to bit positions (keys are
# positions, values are sets of q-grams), to be used for re-identification
# later on
#
q_gram_pos_assign_dict = {}

# Two dictionaries with sets of the identified frequent q-grams as they must or
# cannot occur in a BF. The keys in these dictionaries are record identifiers
# from the encoded data set, while values are sets of q-grams
#
bf_must_have_q_gram_dict =   {}
bf_cannot_have_q_gram_dict = {}

# A set of identified q-grams, once we have a q-gram identified we will not
# consider it in a smaller partiton later in the process
#
identified_q_gram_set = set()

# Set the initial row filter bit array to all 1-bits (i.e use all rows / BFs)
#
row_filter_bit_array = bitarray.bitarray(encode_num_bf)
row_filter_bit_array.setall(1)  # Bit set to 1: use the corresponding BF

# Use a queue of tuples, each consisting of:
# - partition size:       The number of BFs to consider
# - column filter set:    These are the columns not to consider in the pattern
#                         mining approach. Will be empty at beginning, so all
#                         q-grams are considered.
# - row filter bit array: These are the rows (BFs) to consider (1-bits) or not
#                         to consider (0-bits) in the pattern mining approach.
#                         All rows (BFs) are set to 1 at the beginning.
# - the set of q-grams that must be in record q-grams sets (empty at beginning)
# - the set of q-grams that must not be in record q-grams sets (empty at
#                                                               beginning)
#
queue_tuple_list = [(encode_num_bf, set(), row_filter_bit_array, set(), set())]

# Keep the size (number of q-grams) of the most frequent tuple found in each
# iteration as these sizes should correspond to the number of hash functions
#
most_freq_tuple_size_list = []

# As long as there are tuples in the queue process the next tuple
#
iter_num = 0
while (queue_tuple_list != []):
  iter_num += 1

  # Get first tuple from list and remove it from queue (pop it)
  #
  this_part_size, col_filter_set, row_filter_bit_array, \
           must_be_in_rec_q_gram_set, must_not_be_in_rec_q_gram_set = \
                                                 queue_tuple_list.pop(0)
  print
  print 'Iteration %d: ---------------------------------------------' % \
        (iter_num)
  print '  Number of BFs to consider: %d (%.2f%% of all BFs)' % \
        (this_part_size, 100.0*this_part_size/encode_num_bf)
  print '  Column filter set contains %d bit positions (bit positions' % \
        (len(col_filter_set)) + ' not to consider)'
  print '  Row (BF) filter has %d of %d BFs set to 1 (BFs to consider)' % \
          (int(row_filter_bit_array.count(1)), len(row_filter_bit_array))
  print '  Set of q-grams that must be in a record:    ', \
        must_be_in_rec_q_gram_set
  print '  Set of q-grams that must not be in a record:', \
       must_not_be_in_rec_q_gram_set
  print

  # Get the two most frequent q-grams and their counts of occurrence in the
  # plain text data set in the current partition (i.e. with q-gram sets that
  # must be in records or not in records for filtering)
  #
  freq_q_gram_count_list = get_most_freq_other_q_grams(plain_q_gram_dict,
                                                must_be_in_rec_q_gram_set,
                                                must_not_be_in_rec_q_gram_set)
  most_freq_q_gram1, most_freq_q_gram_count1 = freq_q_gram_count_list[0]
  most_freq_q_gram2, most_freq_q_gram_count2 = freq_q_gram_count_list[1]

  print '  Top most frequent q-gram "%s" occurs %d times' % \
        (most_freq_q_gram1, most_freq_q_gram_count1)
  print '  Second most frequent q-gram "%s" occurs %d times' % \
        (most_freq_q_gram2, most_freq_q_gram_count2)
  print

  # If the most frequent q-gram has already been identified in an earlier
  # iteration then don't consider it
  #
  if (most_freq_q_gram1 in identified_q_gram_set):
    print '    *** Most frequent q-gram already identified in an earlier ' + \
          'iteration - no need to re-identify so abort iteration ***'
    print
    continue

  # Calculate the average frequency between the top two q-grams (to be used as
  # minimum count (minimum support) in the pattern mining algorithm), the idea
  # being that this count should only result in columns of the top most
  # frequent q-gram to be included in the final set of selected columns (bit
  # positions).
  #
  avr_top_count = float(most_freq_q_gram_count1+most_freq_q_gram_count2)/2.0

  # To get a suitable minimum count of 1-bits in the Bloom filters, we take the
  # average q-gram count and convert it into a corresponding minimum 1-bit
  # count for the Bloom filter encoded database
  #
  avr_top_q_gram_perc = float(avr_top_count) / plain_num_rec

  # The minimum number of BFs that should have a 1-bit in the columns that
  # possibly can encode the most frequent q-gram
  #
  apriori_bf_min_count = int(math.floor(avr_top_q_gram_perc * encode_num_bf))

  print '  Minimum 1-bit count for BF bit positions: %d' % \
        (apriori_bf_min_count)
  print

  # As stopping criteria check if the difference in counts is large enough
  # (the smaller the difference in count the less clear the pattern mining
  # algorithm will work)
  #
  # Old version: Difference with regard to the full pain-text data set (this
  # version stopped to early)
  #most_freq_count_diff_perc = 100.0*float(most_freq_q_gram_count1 - \
  #                                 most_freq_q_gram_count2) / plain_num_rec
  #
  # Percentage difference between the two most frequent q-grams (in the current
  # partition) relative to each other
  #
  most_freq_count_diff_perc = 100.0*float(most_freq_q_gram_count1 - \
                                   most_freq_q_gram_count2) / avr_top_count

  print '  Percentage difference between two most frequent counts: %.2f%%' \
        % (most_freq_count_diff_perc)

  # Check if the difference is large enough to continue recursive splitting
  #
  #
  if (most_freq_count_diff_perc >= stop_iter_perc):  # Large enough
    print '    Difference large enough (>= %.2f%%) ' % (stop_iter_perc) \
          + 'to continue recursive splitting'
    print

  else:  # Stop the iterative process (do not append new tuples below)
    print '    *** Difference too small to apply Apriori on this partition,',
    print 'abort iteration ***'
    print

    continue  # Go back and process remaining tuples in the queue

  # Run the Apriori pattern mining approach, i.e. find set of bit positions
  # (BF columns) with a minimum count of common 1-bits)
  #
#  freq_bf_bit_pos_dict = gen_freq_bf_bit_positions(encode_bf_bit_pos_list,
#                                                   apriori_bf_min_count,
#                                                   col_filter_set,
#                                                   row_filter_bit_array)

  # Version 2 of the function stores actual BFs not just Hamming weights, so
  # is faster but needs more memory
  #
  freq_bf_bit_pos_dict = gen_freq_bf_bit_positions2(encode_bf_bit_pos_list,
                                                    apriori_bf_min_count,
                                                    col_filter_set,
                                                    row_filter_bit_array)

  # If no frequent bit position tuple found end the iteration
  #
  if (len(freq_bf_bit_pos_dict) == 0):
    print '## Iteration %d:' % (iter_num)
    print '##   No frequent bit position tuple found!'
    print '##'
    continue

  # Get the most frequent tuple of bit positions
  #
  most_freq_pos_tuple, most_freq_count = sorted(freq_bf_bit_pos_dict.items(),
                                                key=lambda t: t[1],
                                                reverse=True)[0]
  print '  Most frequent tuple with positions %s occurs %d times' % \
        (most_freq_pos_tuple, most_freq_count)
  print

  # If this is not the first iteration then check the number of bit positions
  # identified, if much less than the average found in previous iterations
  # then print a warning
  #
  if (most_freq_tuple_size_list != []):
    num_pos_identified = len(most_freq_pos_tuple)

    avr_num_pos_identified = numpy.mean(most_freq_tuple_size_list)

    max_diff = avr_num_pos_identified * CHECK_POS_TUPLE_SIZE_DIFF_PERC / 100.0

    # Check if enough bit positions were identified
    #
    if (num_pos_identified + max_diff < avr_num_pos_identified):
     print '  *** Warning, most frequent tuple does not contain enough ' + \
           'bit positions (%d versus %.1f average so far), abort ' % \
           (num_pos_identified, avr_num_pos_identified) + 'iteration ***'
     print
     continue

  # *************************************************************************
  # TODO PC 20170918: Should we also check the count differences between the
  # two most frequent returned position tuples? If they are too close
  # (less than stop_iter_perc apart, then maybe we do not want to consider
  # this position tuple?
  # *************************************************************************
  # TODO PC 20170919: Or we only consider the q-gram if a single tuple
  # is being returned which is long enough
  # *************************************************************************

  most_freq_tuple_size_list.append(len(most_freq_pos_tuple))

  assert most_freq_count >= apriori_bf_min_count, \
         (most_freq_count, apriori_bf_min_count)

  # Assign the most frequent q-gram from plain text to the selected positions
  #
  for pos in most_freq_pos_tuple:
    pos_q_gram_set = q_gram_pos_assign_dict.get(pos, set())
    pos_q_gram_set.add(most_freq_q_gram1)
    q_gram_pos_assign_dict[pos] = pos_q_gram_set

  # Add the most frequent q-gram to the set of identified q-grams
  #
  identified_q_gram_set.add(most_freq_q_gram1)

  # Count in how many of the selected bit positions does the most frequent
  # q-gram occur (assume true assignment of q-grams to bit positions is known)
  #
  encode_num_bit_pos_with_most_freq_q_gram = 0
  plain_num_bit_pos_with_most_freq_q_gram =  0

  for pos in most_freq_pos_tuple:
    if (most_freq_q_gram1 in encode_true_q_gram_pos_map_dict.get(pos, set())):
      encode_num_bit_pos_with_most_freq_q_gram += 1
    if (most_freq_q_gram1 in plain_true_q_gram_pos_map_dict.get(pos,set())):
      plain_num_bit_pos_with_most_freq_q_gram += 1

  # Print a summary of the iteration and results
  #
  print '## Iteration %d summary:' % (iter_num)
  print '##   Two most frequent q-grams from plain-text and their counts:' \
        + ' ("%s" / %d) and  ("%s" / %d)' % (most_freq_q_gram1,
        most_freq_q_gram_count1, most_freq_q_gram2, most_freq_q_gram_count2)
  print '##   Column filter contains %d bit positions, row bit filter ' % \
        (len(col_filter_set)) + 'has %d of %d 1-bits' % \
        (int(row_filter_bit_array.count(1)), len(row_filter_bit_array))
  print '##   Set of must / must not occurring record q-grams: %s / %s' % \
        (must_be_in_rec_q_gram_set, must_not_be_in_rec_q_gram_set)
  print '##   Most frequent selected set of %d bit positions %s ' % \
        (len(most_freq_pos_tuple), most_freq_pos_tuple) + 'occurs %d times' \
        % (most_freq_count)
  print '##   Most frequent g-gram "%s" occurs in %d of %d selected bit ' \
        % (most_freq_q_gram1, encode_num_bit_pos_with_most_freq_q_gram,
           len(most_freq_pos_tuple)) + 'positions for encode BFs'
  print '##   Most frequent g-gram "%s" occurs in %d of %d selected bit ' \
        % (most_freq_q_gram1, plain_num_bit_pos_with_most_freq_q_gram,
           len(most_freq_pos_tuple)) + 'positions for plain-text BFs'
  print '##'
  print

  # **************************************************************************
  # TODO PC 20170918: We need to check and maybe improve the below assumption
  # **************************************************************************

  # Update the column filter set with the newly assigned columns (we basically
  # assume that once a q-gram has been assigned to a column then do not re-use
  # the column - this is of course not correct
  #
  next_col_filter_set = col_filter_set.union(set(most_freq_pos_tuple))

  # Because q-grams can share bit positions (see birthday paradox probability
  # calculated above), the recursive calls will generate different column
  # filter sets
  #
  print '  Next column filter set:', sorted(next_col_filter_set)
  print

  # Generate the rows (BFs) where all selected columns have a 1-bit (as the
  # intersection of all BF bit positions that have the most frequent q-gram
  # assigned to them)
  #
  sel_bit_row_filter_bit_array = bitarray.bitarray(encode_num_bf)
  sel_bit_row_filter_bit_array.setall(1)

  for pos in most_freq_pos_tuple:
    sel_bit_row_filter_bit_array = sel_bit_row_filter_bit_array & \
                                             encode_bf_bit_pos_list[pos]

  assert int(sel_bit_row_filter_bit_array.count(1)) >= most_freq_count, \
         (int(sel_bit_row_filter_bit_array.count(1)), most_freq_count)

  # Assign the most frequent q-gram to all BFs that have 1-bits in all selected
  # bit positions (as must have q-gram), and as cannot have q-gram to all to
  # all other BFs
  #
  assert len(sel_bit_row_filter_bit_array) == encode_num_bf
  assert len(encode_rec_id_list) == encode_num_bf

  for i in range(encode_num_bf):
    bf_rec_id = encode_rec_id_list[i]

    # A 1-bit means the most frequent q-gram is assumed to occur in a BF
    #
    if (sel_bit_row_filter_bit_array[i] == 1):
      bf_q_gram_set = bf_must_have_q_gram_dict.get(bf_rec_id, set())
      bf_q_gram_set.add(most_freq_q_gram1)
      bf_must_have_q_gram_dict[bf_rec_id] = bf_q_gram_set

    else:  # A 0-bit means the q-gram is assumed not to occur in the BF
      bf_q_gram_set = bf_cannot_have_q_gram_dict.get(bf_rec_id, set())
      bf_q_gram_set.add(most_freq_q_gram1)
      bf_cannot_have_q_gram_dict[bf_rec_id] = bf_q_gram_set

  # Generate the two row filters for the next two pattern mining calls
  #
  next_row_filter_bit_array = row_filter_bit_array & \
                              sel_bit_row_filter_bit_array

  sel_bit_row_filter_bit_array.invert()  # Negate all bits
  next_neg_row_filter_bit_array = row_filter_bit_array & \
                                  sel_bit_row_filter_bit_array
  assert (int(row_filter_bit_array.count(1)) == \
          int(next_row_filter_bit_array.count(1)) + \
          int(next_neg_row_filter_bit_array.count(1))), \
         (int(row_filter_bit_array.count(1)), \
         int(next_row_filter_bit_array.count(1)) + \
         int(next_neg_row_filter_bit_array.count(1)))

  # Add the most frequent q-gram to the set of q-grams that must or must not
  # occur in records for the next two iterations (tuples to be added to the
  # queue)
  #
  next_must_be_in_rec_q_gram_set = \
                 must_be_in_rec_q_gram_set.union(set([most_freq_q_gram1]))
  next_must_not_be_in_rec_q_gram_set = \
                 must_not_be_in_rec_q_gram_set.union(set([most_freq_q_gram1]))

  # **************************************************************************
  # TODO PC 20170918: As stopping criteria we currently use the minimum size
  # of a partition (its number of BFs) - is this good? What are alternatives?
  # For some partitions the re-identification goes totaly wrong - see output
  # of code ("..q-gram "be" occurs in 0 of 8 selected bit positions") - why?
  # Is minimum number of 1-bits in a partition a better indicator?
  # **************************************************************************

  # PC 20170918: this is an old stopping criteria ****************************
  #  if (apriori_bf_min_count < min_part_size):
  #    print '  *** Not enough 1-bits to run Apriori, abort iteration ***'
  #    print
  #    continue
  # **************************************************************************

  # Append two new tuples to queue (one for the sub-set of rows with the most
  # frequent q-gram, the other for rows without the most frequent q-grams)
  #
  # Only add a tuple if its corresponding partition (number of rows to
  # consider) is large enough (larger than min_part_size)
  #
  # In the first tuple, add the new found most frequent q-gram to the set of
  # q-grams that must be in a record.
  # In the second tuple, add it to the set of q-grams that must not be in a
  # record.
  #
  pos_part_size = int(next_row_filter_bit_array.count(1))
  neg_part_size = int(next_neg_row_filter_bit_array.count(1))

  if (pos_part_size >= min_part_size):
    queue_tuple_list.append((pos_part_size, next_col_filter_set,
                             next_row_filter_bit_array,
                             next_must_be_in_rec_q_gram_set, 
                             must_not_be_in_rec_q_gram_set))
    print '  Added positive tuple with %d BFs to the queue' % (pos_part_size)
    print

  if (neg_part_size >= min_part_size):
    queue_tuple_list.append((neg_part_size, next_col_filter_set,
                             next_neg_row_filter_bit_array,
                             must_be_in_rec_q_gram_set, 
                             next_must_not_be_in_rec_q_gram_set))
    print '  Added negative tuple with %d BFs to the queue' % (neg_part_size)
    print

  # Sort the queue according to partition size, with largest partition first
  #
  queue_tuple_list.sort(reverse=True)

# ****************************************************************************
# TODO PC 201709018: from the below list, what is k? (num of hash functions)
# - the mode of the list, or the min, or the max? or the average?
# ****************************************************************************

print 'Size of the most frequent tuples found in all iterations:', \
      most_freq_tuple_size_list

# Take the mode of this list as estimate of the number of hash functions
#
est_num_hash_funct = max(set(most_freq_tuple_size_list), \
                         key=most_freq_tuple_size_list.count)
print '  Estimated number of hash functions: %d' % (est_num_hash_funct)
print

apriori_time = time.time() - start_time

# Final processing of the sets of must have and cannot have q-gram sets per BF:
# Remove cannot have q-grams from sets of must have q-grams
#
for i in range(encode_num_bf):

  if (i in bf_must_have_q_gram_dict) and (i in bf_cannot_have_q_gram_dict):
    bf_rec_id = encode_rec_id_list[i]

    bf_must_have_q_gram_set =   bf_must_have_q_gram_dict[bf_rec_id]
    bf_cannot_have_q_gram_set = bf_cannot_have_q_gram_dict[bf_rec_id]

    # Remove the cannot have q-grams from the must have q-grams
    #
    final_bf_must_have_q_gram_set = \
                      bf_must_have_q_gram_set - bf_cannot_have_q_gram_set
    if (final_bf_must_have_q_gram_set != bf_must_have_q_gram_set):
      bf_must_have_q_gram_dict[bf_rec_id] = final_bf_must_have_q_gram_set

# Output results of Apriori based BF analysis
#
print '#### Apriori BF bit position analysis took %d iterations and %d sec' % \
      (iter_num, apriori_time) + ', %.1f sec per iteration' % \
      (apriori_time/iter_num)
print '####   Encoded data set: ', encode_base_data_set_name
print '####     Attributes used:', encode_attr_name_list
print '####     Number of records and BFs: %d' % (len(encode_q_gram_dict)) + \
      ', time for BF generation: %d sec' % (encode_bf_gen_time)
if (encode_base_data_set_name == plain_base_data_set_name):
  print '####   Plain-text data set: *** Same as build data set ***'
else:
  print '####   Plain-text data set:', plain_base_data_set_name
  print '####     Attributes used:', plain_attr_name_list
  print '####     Number of records and BFs: %d' % \
        (len(plain_q_gram_dict)) + ', time for BF generation: %d sec' % \
        (plain_bf_gen_time)
print '####   Parameters: q=%d, k=%d, bf_len=%d, hash type=%s, BF harden=%s' \
      % (q, num_hash_funct, bf_len, hash_type, bf_harden)
print '####     Most frequent BF pattern occurs %d times, most frequent BF ' \
      % (most_freq_bf_pattern_count) + 'bit pattern occurs %d times' % \
      (most_freq_bf_bit_pos_pattern_count)
print '####'

print '#### Number of q-grams identified: %d (from %d q-grams, %.2f%%)' % \
      (len(identified_q_gram_set), len(plain_q_gram_node_dict),
       100.0*float(len(identified_q_gram_set)) / len(plain_q_gram_node_dict))
print '####'

# For each identified q-gram first get its true bit positions in the encoded
# and the plain-text databases
#
encode_true_q_gram_pos_dict = {}
plain_true_q_gram_pos_dict =  {}

encode_bit_pos_q_gram_reca_list = []
plain_bit_pos_q_gram_reca_list =  []

encode_bit_pos_q_gram_false_pos_list = []  # Also keep track of how many wrong
plain_bit_pos_q_gram_false_pos_list =  []  # positions we identified

for (pos, encode_q_gram_set) in encode_true_q_gram_pos_map_dict.iteritems():
  for q_gram in encode_q_gram_set:
    q_gram_pos_set = encode_true_q_gram_pos_dict.get(q_gram, set())
    q_gram_pos_set.add(pos)
    encode_true_q_gram_pos_dict[q_gram] = q_gram_pos_set
for (pos, plain_q_gram_set) in plain_true_q_gram_pos_map_dict.iteritems():
  for q_gram in plain_q_gram_set:
    q_gram_pos_set = plain_true_q_gram_pos_dict.get(q_gram, set())
    q_gram_pos_set.add(pos)
    plain_true_q_gram_pos_dict[q_gram] = q_gram_pos_set

assigned_q_gram_pos_dict = {}
for (pos, pos_q_gram_set) in q_gram_pos_assign_dict.iteritems():
  for q_gram in pos_q_gram_set:
    q_gram_pos_set = assigned_q_gram_pos_dict.get(q_gram, set())
    q_gram_pos_set.add(pos)
    assigned_q_gram_pos_dict[q_gram] = q_gram_pos_set
print '#### Assignment of BF bit positions to q-grams:'
for (q_gram, pos_set) in sorted(assigned_q_gram_pos_dict.items()):
  print '####   "%s": %s' % (q_gram, str(sorted(pos_set)))

  encode_true_q_gram_set = encode_true_q_gram_pos_dict[q_gram]
  plain_true_q_gram_set =  plain_true_q_gram_pos_dict[q_gram]

  encode_recall = float(len(pos_set.intersection(encode_true_q_gram_set))) / \
                  len(encode_true_q_gram_set)
  plain_recall = float(len(pos_set.intersection(plain_true_q_gram_set))) / \
                 len(plain_true_q_gram_set)

  # Percentage of false identified positions for a q-gram
  #
  encode_false_pos_rate = float(len(pos_set - encode_true_q_gram_set)) / \
                          len(pos_set)
  plain_false_pos_rate = float(len(pos_set - plain_true_q_gram_set)) / \
                          len(pos_set)

  assert (0.0 <= encode_false_pos_rate) and (1.0 >= encode_false_pos_rate), \
         encode_false_pos_rate
  assert (0.0 <= plain_false_pos_rate) and (1.0 >= plain_false_pos_rate), \
         plain_false_pos_rate

  encode_bit_pos_q_gram_reca_list.append(encode_recall)
  plain_bit_pos_q_gram_reca_list.append(plain_recall)

  encode_bit_pos_q_gram_false_pos_list.append(encode_false_pos_rate)
  plain_bit_pos_q_gram_false_pos_list.append(plain_false_pos_rate)

print '####'
print '#### Encoding assignment of q-grams to bit position recall:   ' + \
      '%.2f min / %.2f avr / %.2f max' % \
      (min(encode_bit_pos_q_gram_reca_list),
       numpy.mean(encode_bit_pos_q_gram_reca_list),
       max(encode_bit_pos_q_gram_reca_list))
print '####   Number and percentage of q-grams with recall 1.0: %d / %.2f%%' \
      % (encode_bit_pos_q_gram_reca_list.count(1.0),
         100.0*float(encode_bit_pos_q_gram_reca_list.count(1.0)) / \
         (len(identified_q_gram_set)+0.0001))
print '#### Plain-text assignment of q-grams to bit position recall: ' + \
      '%.2f min / %.2f avr / %.2f max' % \
      (min(plain_bit_pos_q_gram_reca_list),
       numpy.mean(plain_bit_pos_q_gram_reca_list),
       max(plain_bit_pos_q_gram_reca_list))
print '####   Number and percentage of q-grams with recall 1.0: %d / %.2f%%' \
      % (plain_bit_pos_q_gram_reca_list.count(1.0),
         100.0*float(plain_bit_pos_q_gram_reca_list.count(1.0)) / \
         (len(identified_q_gram_set)+0.0001))
print '####'
print '#### Encoding assignment of q-grams to bit position false ' + \
      'positive rate:   %.2f min / %.2f avr / %.2f max' % \
      (min(encode_bit_pos_q_gram_false_pos_list),
       numpy.mean(encode_bit_pos_q_gram_false_pos_list),
       max(encode_bit_pos_q_gram_false_pos_list))
print '####   Number and percentage of q-grams with false positive rate' + \
      ' 0.0: %d / %.2f%%' \
      % (encode_bit_pos_q_gram_false_pos_list.count(0.0),
         100.0*float(encode_bit_pos_q_gram_false_pos_list.count(0.0)) / \
         (len(identified_q_gram_set)+0.0001))
print '#### Plain-text assignment of q-grams to bit position false ' + \
      'positive rate: %.2f min / %.2f avr / %.2f max' % \
      (min(plain_bit_pos_q_gram_false_pos_list),
       numpy.mean(plain_bit_pos_q_gram_false_pos_list),
       max(plain_bit_pos_q_gram_false_pos_list))
print '####   Number and percentage of q-grams with false positive rate' + \
      ' 0.0: %d / %.2f%%' \
      % (plain_bit_pos_q_gram_false_pos_list.count(0.0),
         100.0*float(plain_bit_pos_q_gram_false_pos_list.count(0.0)) / \
         (len(identified_q_gram_set)+0.0001))
print '####'

# Calculate the precision of the assignment of q-grams to bit positions
#
encode_q_gram_to_bit_pos_assign_prec_list = []
plain_q_gram_to_bit_pos_assign_prec_list =  []

encode_total_num_correct = 0  # Also count how many assignments of q-grams to
encode_total_num_wrong =   0  # bit positions are wrong and how many correct
plain_total_num_correct =  0
plain_total_num_wrong =    0

print '#### Assignment of q-grams to BF bit positions:'
for (pos, pos_q_gram_set) in sorted(q_gram_pos_assign_dict.items()):
  q_gram_set_str_list = []  # Strings to be printed

  encode_pos_corr = 0  # Correctly assigned q-grams to this bit position
  plain_pos_corr =  0

  for q_gram in pos_q_gram_set:
    if (q_gram in encode_true_q_gram_pos_map_dict.get(pos, set())):
      assign_str = 'encode correct'
      encode_pos_corr += 1
      encode_total_num_correct += 1
    else:
      assign_str = 'encode wrong'
      encode_total_num_wrong += 1
    if (same_data_attr_flag == False):  # Check analysis BF
      if (q_gram in plain_true_q_gram_pos_map_dict.get(pos, set())):
        assign_str += ', plain-text correct'
        plain_pos_corr += 1
        plain_total_num_correct += 1
      else:
        assign_str += ', plain-text wrong'
        plain_total_num_wrong += 1
    else:  # Encode and plain-text data sets are the same
      if (q_gram in encode_true_q_gram_pos_map_dict.get(pos, set())):
        assign_str = 'plain-text correct'
        plain_pos_corr += 1
        plain_total_num_correct += 1
      else:
        assign_str = 'plain-text wrong'
        plain_total_num_wrong += 1

    q_gram_set_str_list.append('"%s" (%s)' % (q_gram, assign_str))

  encode_pos_proc = float(encode_pos_corr) / len(pos_q_gram_set)
  plain_pos_proc =  float(plain_pos_corr) / len(pos_q_gram_set)

  encode_q_gram_to_bit_pos_assign_prec_list.append(encode_pos_proc)
  plain_q_gram_to_bit_pos_assign_prec_list.append(plain_pos_proc)

  print '####   %3d: %s' % (pos, ', '.join(q_gram_set_str_list))

print '####'
print '#### Encoding q-gram to bit position assignment precision:   ' + \
      '%.2f min / %.2f avr / %.2f max' % \
      (min(encode_q_gram_to_bit_pos_assign_prec_list),
       numpy.mean(encode_q_gram_to_bit_pos_assign_prec_list),
       max(encode_q_gram_to_bit_pos_assign_prec_list))
print '####   Number and percentage of positions with precison 1.0: ' + \
      '%d / %.2f%%' % (encode_q_gram_to_bit_pos_assign_prec_list.count(1.0),
         100.0*float(encode_q_gram_to_bit_pos_assign_prec_list.count(1.0)) / \
         (len(q_gram_pos_assign_dict)+0.0001))
print '#### Plain-text q-gram to bit position assignment precision: ' + \
      '%.2f min / %.2f avr / %.2f max' % \
      (min(plain_q_gram_to_bit_pos_assign_prec_list),
       numpy.mean(plain_q_gram_to_bit_pos_assign_prec_list),
       max(plain_q_gram_to_bit_pos_assign_prec_list))
print '####   Number and percentage of positions with precison 1.0: ' + \
      '%d / %.2f%%' % (plain_q_gram_to_bit_pos_assign_prec_list.count(1.0),
         100.0*float(plain_q_gram_to_bit_pos_assign_prec_list.count(1.0)) / \
         (len(q_gram_pos_assign_dict)+0.0001))
print '#### Encoding total number of correct and wrong assignments:  ' + \
      '%d / %d (%.2f%% correct)' % (encode_total_num_correct,
         encode_total_num_wrong, 100.0*float(encode_total_num_correct) / \
         (encode_total_num_correct + encode_total_num_wrong + 0.0001))
print '#### Plain-text total number of correct and wrong assignments: ' + \
      '%d / %d (%.2f%% correct)' % (plain_total_num_correct,
         plain_total_num_wrong, 100.0*float(plain_total_num_correct) / \
         (plain_total_num_correct + plain_total_num_wrong + 0.0001))
print '#### '+'-'*80
print '####'
print

# Calculate statistics and quality of the must have and cannot have q-gram
# sets assigned to BFs
#
bf_must_have_set_size_list =   []
bf_cannot_have_set_size_list = []

for q_gram_set in bf_must_have_q_gram_dict.itervalues():
  bf_must_have_set_size_list.append(len(q_gram_set))
for q_gram_set in bf_cannot_have_q_gram_dict.itervalues():
  bf_cannot_have_set_size_list.append(len(q_gram_set))

print '#### Summary of q-gram sets assigned to BFs:'
print '####  %d of %d BF have must have q-grams assigned to them' % \
      (len(bf_must_have_set_size_list), encode_num_bf)
print '####    Minimum, average and maximum number of q-grams assigned: ' + \
      '%d / %.2f / %d' % (min(bf_must_have_set_size_list),
                          numpy.mean(bf_must_have_set_size_list),
                          max(bf_must_have_set_size_list))
print '####  %d of %d BF have cannot have q-grams assigned to them' % \
      (len(bf_cannot_have_set_size_list), encode_num_bf)
print '####    Minimum, average and maximum number of q-grams assigned: ' + \
      '%d / %.2f / %d' % (min(bf_cannot_have_set_size_list),
                          numpy.mean(bf_cannot_have_set_size_list),
                          max(bf_cannot_have_set_size_list))
print '####'

# Calculate the quality of the identified must / cannot q-gram sets as:
# - precision of must have q-grams (how many of those identified are in a BF)
# - precision of cannot have q-grams (how many of those identified are not in
#   a BF)
#
bf_must_have_prec_list =   []
bf_cannot_have_prec_list = []

for (bf_rec_id, q_gram_set) in bf_must_have_q_gram_dict.iteritems():
  true_q_gram_set = encode_q_gram_dict[bf_rec_id]
  must_have_prec = float(len(q_gram_set & true_q_gram_set)) / len(q_gram_set)

  bf_must_have_prec_list.append(must_have_prec)

for (bf_rec_id, q_gram_set) in bf_cannot_have_q_gram_dict.iteritems():
  true_q_gram_set = encode_q_gram_dict[bf_rec_id]

  cannot_have_prec = 1.0 - float(len(q_gram_set & true_q_gram_set)) / \
                     len(q_gram_set)

  bf_cannot_have_prec_list.append(cannot_have_prec)

print '#### Precision of q-gram sets assigned to BFs:'
print '####   Must have q-gram sets minimum, average, maximum precision: ' + \
      '%.2f / %.2f / %.2f' % (min(bf_must_have_prec_list),
                              numpy.mean(bf_must_have_prec_list),
                              max(bf_must_have_prec_list))
print '####     Ratio of BFs with must have precision 1.0: %.3f' % \
      (float(bf_must_have_prec_list.count(1.0)) / \
       (len(bf_must_have_prec_list)+0.0001))
print '####   Cannot have q-gram sets minimum, average, maximum precision: ' \
      + '%.2f / %.2f / %.2f' % (min(bf_cannot_have_prec_list),
                                numpy.mean(bf_cannot_have_prec_list),
                                max(bf_cannot_have_prec_list))
print '####     Ratio of BFs with cannot have precision 1.0: %.3f' % \
      (float(bf_cannot_have_prec_list.count(1.0)) / \
       (len(bf_cannot_have_prec_list)+0.0001))
print '####'
print

# -----------------------------------------------------------------------------
# Step 4: Re-identify plain-text values based on q-grams assigned to bit
#         positions.

if (re_id_method == 'all'):
  re_id_method_list = ['bf_tuple', 'q_gram_tuple', 'bf_q_gram_tuple',
                       'set_inter']
else:
  re_id_method_list = [re_id_method]

for re_id_method in re_id_method_list:

  start_time = time.time()

  if (re_id_method == 'set_inter'):
    reid_res_tuple = re_identify_attr_val_setinter(bf_must_have_q_gram_dict,
                                                   bf_cannot_have_q_gram_dict,
                                                   plain_q_gram_attr_val_dict,
                                                   encode_rec_val_dict,
                                                   max_num_many)
  elif (re_id_method == 'apriori'):
    reid_res_tuple = re_identify_attr_val_apriori(bf_must_have_q_gram_dict,
                                                  bf_cannot_have_q_gram_dict,
                                                  plain_q_gram_attr_val_dict,
                                                  encode_rec_val_dict,
                                                  max_num_many)
  elif (re_id_method == 'q_gram_tuple'):

    # First get sets of bit positions per frequent q-gram
    #
    all_identified_q_gram_pos_dict, corr_identified_q_gram_pos_dict = \
              gen_freq_q_gram_bit_post_dict(q_gram_pos_assign_dict,
                                            encode_true_q_gram_pos_map_dict)

    all_bf_q_gram_rec_id_dict = \
                get_matching_q_gram_sets(all_identified_q_gram_pos_dict,
                                         encode_bf_dict,
                                         plain_q_gram_attr_val_dict,
                                         plain_attr_val_rec_id_dict,
                                         bf_cannot_have_q_gram_dict, bf_len)

    reid_res_tuple = calc_reident_accuracy(all_bf_q_gram_rec_id_dict,
                                           encode_rec_val_dict,
                                           plain_rec_val_dict, max_num_many)

  elif (re_id_method == 'bf_tuple'):
    # First get sets of bit positions per frequent q-gram
    #
    all_identified_q_gram_pos_dict, corr_identified_q_gram_pos_dict = \
              gen_freq_q_gram_bit_post_dict(q_gram_pos_assign_dict,
                                            encode_true_q_gram_pos_map_dict)

    all_bf_q_gram_rec_id_dict = \
                get_matching_bf_sets(all_identified_q_gram_pos_dict,
                                     encode_bf_dict,
                                     plain_attr_val_rec_id_dict,
                                     bf_must_have_q_gram_dict,
                                     bf_cannot_have_q_gram_dict, bf_len)

    reid_res_tuple = calc_reident_accuracy(all_bf_q_gram_rec_id_dict,
                                           encode_rec_val_dict,
                                           plain_rec_val_dict, max_num_many)

  elif (re_id_method == 'bf_q_gram_tuple'):
    # First get sets of bit positions per frequent q-gram
    #
    all_identified_q_gram_pos_dict, corr_identified_q_gram_pos_dict = \
              gen_freq_q_gram_bit_post_dict(q_gram_pos_assign_dict,
                                            encode_true_q_gram_pos_map_dict)

    all_bf_q_gram_rec_id_dict = \
                get_matching_bf_q_gram_sets(all_identified_q_gram_pos_dict,
                                            encode_bf_dict, plain_q_gram_dict,
                                            bf_cannot_have_q_gram_dict, bf_len)

    reid_res_tuple = calc_reident_accuracy(all_bf_q_gram_rec_id_dict,
                                           encode_rec_val_dict,
                                           plain_rec_val_dict, max_num_many)


  elif (re_id_method == 'none'):  # Don't attempt re-identification
    reid_res_tuple = (0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, {})

  else:
    print '*** Should not happen:', re_id_method, '***'
    sys.exit()

  reid_time = time.time() - start_time

  num_no_guess =               reid_res_tuple[0]
  num_too_many_guess =         reid_res_tuple[1]
  num_1_1_guess =              reid_res_tuple[2]
  num_corr_1_1_guess =         reid_res_tuple[3]
  num_part_1_1_guess =         reid_res_tuple[4]
  num_1_m_guess =              reid_res_tuple[5]
  num_corr_1_m_guess =         reid_res_tuple[6]
  num_part_1_m_guess =         reid_res_tuple[7]
  acc_part_1_1_guess =         reid_res_tuple[8]
  acc_part_1_m_guess =         reid_res_tuple[9]
  corr_reid_bf_attr_val_dict = reid_res_tuple[10]

  # ---------------------------------------------------------------------------
  # Print summary results
  #
  today_time_str = time.strftime("%Y%m%d %H:%M:%S", time.localtime())

  print '#### ---------------------------------------------'
  print '#### Run at:', today_time_str
  print '####  ', auxiliary.get_memory_usage()
  print '####   Time used for building (load and q-gram generation / BF ' + \
        'generation): %.2f / %.2f sec' % (encode_load_time, encode_bf_gen_time)
  print '####   Time for analysis (Apriori) and re-identification: ' + \
        '%.2f / %.2f sec' % (apriori_time, reid_time)
  print '#### Encode data set: %s' % (encode_base_data_set_name)
  print '####   Number of records: %d' % (len(encode_q_gram_dict))
  print '####   Attribute(s) used: %s' % (str(encode_attr_name_list))
  if (same_data_attr_flag == False):
    print '#### Analysis data set: %s' % (plain_base_data_set_name)
    print '####   Number of records: %d' % (len(plain_q_gram_dict))
    print '####   Attribute(s) used: %s' % (str(plain_attr_name_list))
  print '####'

  print '#### q: %d' % (q)
  print '#### BF len: %d' % (bf_len)
  print '#### Num hash funct: %d' % (num_hash_funct)
  print '#### Hashing type: %s' % \
        ({'dh':'Double hashing', 'rh':'Random hashing'}[hash_type])
  print '#### BF hardening: %s' % (bf_harden)
  print '#### Stop iteration minimum percentage difference: %.2f' % \
        (stop_iter_perc)
  print '#### Stop iteration minimum partition size: %d' % (min_part_size)
  print '####'

  print '#### Number of freqent q-grams identified: %d' % \
        (len(identified_q_gram_set))
  print '#### Estimate of number of hash functions: %d' % (est_num_hash_funct)

  print '#### Encoding assignment of q-grams to bit position recall:   ' + \
        '%.2f min / %.2f avr / %.2f max' % \
        (min(encode_bit_pos_q_gram_reca_list),
         numpy.mean(encode_bit_pos_q_gram_reca_list),
         max(encode_bit_pos_q_gram_reca_list))
  print '####   Number and percentage of q-grams with recall 1.0: %d / %.2f%%' \
        % (encode_bit_pos_q_gram_reca_list.count(1.0),
           100.0*float(encode_bit_pos_q_gram_reca_list.count(1.0)) / \
           (len(identified_q_gram_set)+0.0001))
  print '#### Plain-text assignment of q-grams to bit position recall: ' + \
        '%.2f min / %.2f avr / %.2f max' % \
        (min(plain_bit_pos_q_gram_reca_list),
         numpy.mean(plain_bit_pos_q_gram_reca_list),
         max(plain_bit_pos_q_gram_reca_list))
  print '####   Number and percentage of q-grams with recall 1.0: %d / %.2f%%' \
        % (plain_bit_pos_q_gram_reca_list.count(1.0),
           100.0*float(plain_bit_pos_q_gram_reca_list.count(1.0)) / \
           (len(identified_q_gram_set)+0.0001))
  print '####'

  print '#### Encoding q-gram to bit position assignment precision:   ' + \
        '%.2f min / %.2f avr / %.2f max' % \
        (min(encode_q_gram_to_bit_pos_assign_prec_list),
         numpy.mean(encode_q_gram_to_bit_pos_assign_prec_list),
         max(encode_q_gram_to_bit_pos_assign_prec_list))
  print '####   Number and percentage of positions with precison 1.0: ' + \
        '%d / %.2f%%' % (encode_q_gram_to_bit_pos_assign_prec_list.count(1.0),
           100.0*float(encode_q_gram_to_bit_pos_assign_prec_list.count(1.0)) / \
           (len(q_gram_pos_assign_dict)+0.0001))
  print '#### Plain-text q-gram to bit position assignment precision: ' + \
        '%.2f min / %.2f avr / %.2f max' % \
        (min(plain_q_gram_to_bit_pos_assign_prec_list),
         numpy.mean(plain_q_gram_to_bit_pos_assign_prec_list),
         max(plain_q_gram_to_bit_pos_assign_prec_list))
  print '####   Number and percentage of positions with precison 1.0: ' + \
        '%d / %.2f%%' % (plain_q_gram_to_bit_pos_assign_prec_list.count(1.0),
           100.0*float(plain_q_gram_to_bit_pos_assign_prec_list.count(1.0)) / \
           (len(q_gram_pos_assign_dict)+0.0001))
  print '####'

  print '#### Encoding total number of correct and wrong assignments:  ' + \
        '%d / %d (%.2f%% correct)' % (encode_total_num_correct,
           encode_total_num_wrong, 100.0*float(encode_total_num_correct) / \
           (encode_total_num_correct + encode_total_num_wrong))
  print '#### Plain-text total number of correct and wrong assignments: ' + \
        '%d / %d (%.2f%% correct)' % (plain_total_num_correct,
           plain_total_num_wrong, 100.0*float(plain_total_num_correct) / \
           (plain_total_num_correct + plain_total_num_wrong))
  print '####'

  print '#### Must have minimum, average and maximum number of q-grams ' + \
        'assigned: %d / %.2f / %d' % (min(bf_must_have_set_size_list),
                                      numpy.mean(bf_must_have_set_size_list),
                                      max(bf_must_have_set_size_list))
  print '#### Cannot have minimum, average and maximum number of q-grams ' + \
        'assigned: %d / %.2f / %d' % (min(bf_cannot_have_set_size_list),
                                      numpy.mean(bf_cannot_have_set_size_list),
                                      max(bf_cannot_have_set_size_list))
  print '####'
  print '#### Must have q-gram sets minimum, average, maximum precision: ' + \
        '%.2f / %.2f / %.2f' % (min(bf_must_have_prec_list),
                                numpy.mean(bf_must_have_prec_list),
                                max(bf_must_have_prec_list))
  print '####   Ratio of BFs with must have precision 1.0: %.3f' % \
        (float(bf_must_have_prec_list.count(1.0)) / \
         (len(bf_must_have_prec_list)+0.0001))
  print '#### Cannot have q-gram sets minimum, average, maximum precision: ' + \
        '%.2f / %.2f / %.2f' % (min(bf_cannot_have_prec_list),
                                numpy.mean(bf_cannot_have_prec_list),
                                max(bf_cannot_have_prec_list))
  print '####   Ratio of BFs with cannot have precision 1.0: %.3f' % \
        (float(bf_cannot_have_prec_list.count(1.0)) / \
         (len(bf_cannot_have_prec_list)+0.0001))
  print '####'

  print '#### Re-identification method:', re_id_method
  print '#### Num no guesses: %d' % (num_no_guess)
  print '####   Num > %d guesses: %d' % (max_num_many, num_too_many_guess)
  print '####   Num 2 to %d guesses: %d' % (max_num_many, num_1_m_guess)
  print '####     Num correct 2 to %d guesses: %d' % \
        (max_num_many, num_corr_1_m_guess)
  if (num_part_1_m_guess > 0):
    print '####     Num partially correct 2 to %d guesses: %d' % \
          (max_num_many, num_part_1_m_guess) + \
          ' (average accuracy of common tokens: %.2f)' % (acc_part_1_m_guess)
  else:
    print '####     Num partially correct 2 to %d guesses: 0' % (max_num_many)
  print '#### Num 1-1 guesses: %d' % (num_1_1_guess)
  print '####   Num correct 1-1 guesses: %d' % (num_corr_1_1_guess)
  if (num_part_1_1_guess > 0):
    print '####   Num partially correct 1-1 guesses: %d' % \
          (num_part_1_1_guess) + \
          ' (average accuracy of common tokens: %.2f)' % (acc_part_1_1_guess)
  else:
    print '####   Num partially correct 1-1 guesses: 0'
  print '####'
  print

  res_list = [today_time_str, encode_base_data_set_name,
              len(encode_q_gram_dict),
              str(encode_attr_name_list), plain_base_data_set_name,
              len(plain_q_gram_dict), str(plain_attr_name_list),
              #
              encode_load_time, encode_bf_gen_time,
              #
              q, bf_len, num_hash_funct, hash_type, bf_harden,
              #
              stop_iter_perc, min_part_size,
              #
              apriori_time, reid_time,
              #
              len(identified_q_gram_set), est_num_hash_funct,
              #
              # Assignment of q-grams to bit position recall:
              min(encode_bit_pos_q_gram_reca_list),
              numpy.mean(encode_bit_pos_q_gram_reca_list),
              max(encode_bit_pos_q_gram_reca_list),
              100.0*float(encode_bit_pos_q_gram_reca_list.count(1.0)) / \
                  (len(identified_q_gram_set)+0.0001),
              min(plain_bit_pos_q_gram_reca_list),
              numpy.mean(plain_bit_pos_q_gram_reca_list),
              max(plain_bit_pos_q_gram_reca_list),
              100.0*float(plain_bit_pos_q_gram_reca_list.count(1.0)) / \
                  (len(identified_q_gram_set)+0.0001),
              #
              # Q-gram to bit position assignment precision
              min(encode_q_gram_to_bit_pos_assign_prec_list),
              numpy.mean(encode_q_gram_to_bit_pos_assign_prec_list),
              max(encode_q_gram_to_bit_pos_assign_prec_list),
              100.0* \
                float(encode_q_gram_to_bit_pos_assign_prec_list.count(1.0)) / \
                (len(q_gram_pos_assign_dict)+0.0001),
              min(plain_q_gram_to_bit_pos_assign_prec_list),
              numpy.mean(plain_q_gram_to_bit_pos_assign_prec_list),
              max(plain_q_gram_to_bit_pos_assign_prec_list),
              100.0* \
                float(plain_q_gram_to_bit_pos_assign_prec_list.count(1.0)) / \
                (len(q_gram_pos_assign_dict)+0.0001),
              #
              # Total number of correct and wrong assignments:
              encode_total_num_correct,
              encode_total_num_wrong, 100.0*float(encode_total_num_correct) / \
              (encode_total_num_correct + encode_total_num_wrong + 0.0001),
              plain_total_num_correct,
              plain_total_num_wrong, 100.0*float(plain_total_num_correct) / \
              (plain_total_num_correct + plain_total_num_wrong + 0.0001),
              #
              # Must have and cannot have numbers of q-grams
              min(bf_must_have_set_size_list),
              numpy.mean(bf_must_have_set_size_list),
              max(bf_must_have_set_size_list),
              min(bf_cannot_have_set_size_list),
              numpy.mean(bf_cannot_have_set_size_list),
              max(bf_cannot_have_set_size_list),
              #
              # Must have q-gram sets precision
              min(bf_must_have_prec_list),
              numpy.mean(bf_must_have_prec_list),
              max(bf_must_have_prec_list),
              float(bf_must_have_prec_list.count(1.0)) / \
                (len(bf_must_have_prec_list)+0.0001),
              #
              # Cannot have q-gram sets precision
              min(bf_cannot_have_prec_list),
              numpy.mean(bf_cannot_have_prec_list),
              max(bf_cannot_have_prec_list),
              float(bf_cannot_have_prec_list.count(1.0)) / \
                (len(bf_cannot_have_prec_list)+0.0001),
              #
              # Re-identification quality
              re_id_method, max_num_many,
              num_no_guess, num_too_many_guess, num_1_1_guess,
              num_corr_1_1_guess, num_part_1_1_guess, num_1_m_guess,
              num_corr_1_m_guess, num_part_1_m_guess,
              acc_part_1_1_guess, acc_part_1_m_guess
             ]

  # Generate header line with column names
  #
  header_list = ['today_time_str', 'encode_data_set_name', 'encode_num_rec',
                 'encode_used_attr', 'plain_data_set_name', 'plain_num_rec',
                 'plain_used_attr',
                 #
                 'encode_load_time', 'encode_bf_gen_time',
                 #
                 'q', 'bf_len', 'num_hash_funct', 'hash_type', 'bf_harden',
                 #
                 'stop_iter_perc', 'min_part_size',
                 #
                 'apriori_time', 'reid_time',
                 #
                 'num_identified_q_gram', 'est_num_hash_funct',
                 #
                 'encode_min_bit_poss_assign_reca',
                 'encode_avr_bit_poss_assign_reca',
                 'encode_max_bit_poss_assign_reca',
                 'encode_perc_1_bit_poss_assign_reca',
                 'plain_min_bit_poss_assign_reca',
                 'plain_avr_bit_poss_assign_reca',
                 'plain_max_bit_poss_assign_reca',
                 'plain_perc_1_bit_poss_assign_reca',
                 #
                 'encode_min_q_gram_poss_assign_prec',
                 'encode_avr_q_gram_poss_assign_prec',
                 'encode_max_q_gram_poss_assign_prec',
                 'encode_perc_1_q_gram_poss_assign_prec',
                 'plain_min_q_gram_poss_assign_prec',
                 'plain_avr_q_gram_poss_assign_prec',
                 'plain_max_q_gram_poss_assign_prec',
                 'plain_perc_1_q_gram_poss_assign_prec',
                 #
                 'encode_total_num_corr_assign',
                 'encode_total_num_wrong_assign',
                 'encode_perc_corr_assign',
                 'plain_total_num_corr_assign', 'plain_total_num_wrong_assign',
                 'plain_perc_corr_assign',
                 #
                 'must_have_min_num_q_gram', 'must_have_avr_num_q_gram',
                 'must_have_max_num_q_gram',
                 'cannot_have_min_num_q_gram', 'cannot_have_avr_num_q_gram',
                 'cannot_have_max_num_q_gram',
                 #
                 'must_have_q_gram_min_prec', 'must_have_q_gram_avr_prec',
                 'must_have_q_gram_max_prec', 'must_have_q_gram_perc_1_prec',
                 'cannot_have_q_gram_min_prec', 'cannot_have_q_gram_avr_prec',
                 'cannot_have_q_gram_max_prec',
                 'cannot_have_q_gram_perc_1_prec',
                 #
                 're_id_method', 'max_num_many', 'num_no_guess',
                 'num_too_many_guess', 'num_1_1_guess', 'num_corr_1_1_guess',
                 'num_part_1_1_guess', 'num_1_m_guess', 'num_corr_1_m_guess',
                 'num_part_1_m_guess', 'acc_part_1_1_guess',
                 'acc_part_1_m_guess'
                ]

  # Check if the result file exists, if it does append, otherwise create
  #
  if (not os.path.isfile(res_file_name)):
    csv_writer = csv.writer(open(res_file_name, 'w'))

    csv_writer.writerow(header_list)

    print 'Created new result file:', res_file_name

  else:  # Append results to an existing file
    csv_writer = csv.writer(open(res_file_name, 'a'))

    print 'Append results to file:', res_file_name

  csv_writer.writerow(res_list)

  print '  Written result line:'
  print ' ', res_list

  assert len(res_list) == len(header_list)

  print
  print '='*80
  print

print 'End.'
