
# Script to run the column based BF attack program on different data sets and
# with various parameter settings
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

import os
import sys

run_mode = sys.argv[1]
assert run_mode in ['run','number','show'], run_mode

# -----------------------------------------------------------------------------
#
# Attribute names: voter_id, first_name, middle_name, last_name, age,
#   gender, street_address, city, state, zip_code, full_phone_num

# Name of encode and plain-text file names
#
data_set_file_name_pair_list = \
  [('data/ncvoter-20140619-temporal-balanced-ratio-1to1-a.csv.gz',
    'data/ncvoter-20140619-temporal-balanced-ratio-1to1-b.csv.gz'),
  ]

# Pairs of attribute lists for encode and plain-text data sets
#
attr_pair_list = [('[1]','[1]'), ('[3]','[3]'), ('[6]','[6]'), ('[7]','[7]'),
                  ('[1,3]','[1,3]'), ('[1,6]','[1,6]'), ('[1,7]','[1,7]'),
                  ('[1,3,6]','[1,3,6]'), ('[1,3,7]','[1,3,7]'),
                  ('[1,3,6,7]','[1,3,6,7]')
                 ]

encode_rec_id_col =       0
encode_col_sep =          ','
encode_header_line_flag = 'True'

plain_rec_id_col =       0
plain_col_sep =          ','
plain_header_line_flag = 'True'

q_list = [2]

bf_len_list = [1000]

num_hash_funct_list = [10, 15, 20]

hash_type_list = ['DH', 'RH']

bf_harden_list = ['none']  # ['none', 'balance', 'fold']

stop_iter_perc_list = [1.0, 5.0]

min_part_size_list = [10000, 2000]

max_num_many_list = [10]

re_id_method_list = ['bf_tuple']  # ['all']

# -----------------------------------------------------------------------------

run_call_list = []  # System calls to be run

# Loop over the different parameter settings
#
for (encode_file_name, plain_file_name) in data_set_file_name_pair_list:

  for num_hash_funct in num_hash_funct_list:

    for q in q_list:

      for bf_len in bf_len_list:

        for hash_type in hash_type_list:

          for bf_harden in bf_harden_list:

            for stop_iter_perc in stop_iter_perc_list:

              for min_part_size in min_part_size_list:

                for max_num_many in  max_num_many_list:

                  for re_id_method in re_id_method_list:

                    for attr_pair in attr_pair_list:

                      encode_attr_str, plain_attr_str = \
                                            attr_pair[0], attr_pair[1]

                      sys_call_str = 'python bf_attack-col-pattern.py ' + \
                                     '%d %s %d %d %s ' % \
                                     (q, hash_type, num_hash_funct, bf_len, \
                                      bf_harden)
                      sys_call_str += '%.1f %d ' % \
                                      (stop_iter_perc, min_part_size)
                      sys_call_str += '%s %s %s %s %s ' % \
                                      (encode_file_name, encode_rec_id_col, \
                                       encode_col_sep, \
                                       encode_header_line_flag, \
                                       encode_attr_str)
                      sys_call_str += '%s %s %s %s %s ' % \
                                      (plain_file_name, plain_rec_id_col, \
                                       plain_col_sep, \
                                       plain_header_line_flag, \
                                       plain_attr_str)
                      sys_call_str += '%d %s' % (max_num_many, re_id_method)

                      run_call_list.append(sys_call_str)

print 'Number of experiments:', len(run_call_list)

if (run_mode == 'number'):
  sys.exit()  # Only show number

for sys_call_str in run_call_list:

  if (run_mode == 'show'):
    print '='*100
    print
    print sys_call_str
    print

  elif (run_mode == 'run'):  # Else only show the calls
    os.system(sys_call_str)

  else:
    print '** Wrong:', run_mode
    sys.exit()

# End.
