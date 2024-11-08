import re
import argparse

malloc_pattern = r"malloc\s+(0x[0-9a-fA-F]+),\s*size\s+(\d+)"
free_pattern = r"free\s+(0x[0-9a-fA-F]+)"

key_offset = -1
value_offset = -1

class Node:
  def __init__(self, idx, name, addrs = [], all_addr_corresponding_malloc_idx = []):
    self.idx = idx
    self.name = name
    self.addrs = addrs
    self.all_addr_corresponding_malloc_idx = all_addr_corresponding_malloc_idx


      
def get_node_type(line):
  match = re.search(r"node type: (\d+)", line)

  if match:
    node_type = int(match.group(1))
    return node_type
  else:
    print("No match found.")
      
      
def get_node_idx(line):
  match = re.search(r"node idx: (\d+)", line)

  if match:
    node_idx = int(match.group(1))
    return node_idx
  else:
    print("No match found.")
      
      
def get_node_func_name(line):
  match = re.search(r"kernel_func\.name mangled = (.+)", line)

  if match:
    func_name = match.group(1)
    return func_name
  else:
    print("No match found.")

def process_memset_kernel(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 1]
  pattern = r"memset_params.dst = (0x[\da-fA-F]+)"
  match = re.search(pattern, addr_line)
  if match:
    address = match.group(1)
    addrs.append(address)
  else:
    print("No match found.")
    exit(-1)
    
  return addrs
  

def process_ampere_kernel(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 2]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  first_input_matrix_addr = ''.join(reversed_numbers)
  first_input_matrix_addr = "0x" + first_input_matrix_addr.lstrip('0')
  addrs.append(first_input_matrix_addr)
  reversed_numbers = numbers[-8:][::-1]
  second_input_matrix_addr = ''.join(reversed_numbers)
  second_input_matrix_addr = "0x" + second_input_matrix_addr.lstrip('0')
  addrs.append(second_input_matrix_addr)
  
  addr_line = all_lines[line_no + 3]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  output_matrix_addr = ''.join(reversed_numbers)
  output_matrix_addr = "0x" + output_matrix_addr.lstrip('0')
  addrs.append(output_matrix_addr)
  
  addr_line = all_lines[line_no + 9]
  numbers = addr_line.split()
  reversed_numbers = numbers[-8:][::-1]
  tmp_matrix_addr = ''.join(reversed_numbers)
  tmp_matrix_addr = "0x" + tmp_matrix_addr.lstrip('0')
  addrs.append(tmp_matrix_addr)
  
  return addrs


def process_sm80_kernel(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 2]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  reversed_numbers = numbers[-8:][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 3]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 5]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  return addrs


def process_ampere_kernel_208(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 2]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  first_input_matrix_addr = ''.join(reversed_numbers)
  first_input_matrix_addr = "0x" + first_input_matrix_addr.lstrip('0')
  addrs.append(first_input_matrix_addr)
  reversed_numbers = numbers[-8:][::-1]
  second_input_matrix_addr = ''.join(reversed_numbers)
  second_input_matrix_addr = "0x" + second_input_matrix_addr.lstrip('0')
  addrs.append(second_input_matrix_addr)
  
  addr_line = all_lines[line_no + 3]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  output_matrix_addr = ''.join(reversed_numbers)
  output_matrix_addr = "0x" + output_matrix_addr.lstrip('0')
  addrs.append(output_matrix_addr)
  
  addr_line = all_lines[line_no + 9]
  numbers = addr_line.split()
  reversed_numbers = numbers[-8:][::-1]
  tmp_matrix_addr = ''.join(reversed_numbers)
  tmp_matrix_addr = "0x" + tmp_matrix_addr.lstrip('0')
  addrs.append(tmp_matrix_addr)
  
  addr_line = all_lines[line_no + 13]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  tmp_matrix_addr = ''.join(reversed_numbers)
  tmp_matrix_addr = "0x" + tmp_matrix_addr.lstrip('0')
  addrs.append(tmp_matrix_addr)
  
  return addrs

def process_paged_attention_v1_kernel(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 2]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  out_addr = ''.join(reversed_numbers)
  out_addr = "0x" + out_addr.lstrip('0')
  addrs.append(out_addr)
  
  addr_line = all_lines[line_no + 5]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  q_addr = ''.join(reversed_numbers)
  q_addr = "0x" + q_addr.lstrip('0')
  addrs.append(q_addr)
  
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  k_cache_addr = ''.join(reversed_numbers)
  k_cache_addr = "0x" + k_cache_addr.lstrip('0')
  addrs.append(k_cache_addr)
  
  addr_line = all_lines[line_no + 11]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  v_cache_addr = ''.join(reversed_numbers)
  v_cache_addr = "0x" + v_cache_addr.lstrip('0')
  addrs.append(v_cache_addr)
  
  addr_line = all_lines[line_no + 20]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  block_tables_addr = ''.join(reversed_numbers)
  block_tables_addr = "0x" + block_tables_addr.lstrip('0')
  addrs.append(block_tables_addr)
  
  addr_line = all_lines[line_no + 23]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  context_lens_addr = ''.join(reversed_numbers)
  context_lens_addr = "0x" + context_lens_addr.lstrip('0')
  addrs.append(context_lens_addr)
  
  addr_line = all_lines[line_no + 29]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  context_lens_addr = ''.join(reversed_numbers)
  context_lens_addr = "0x" + context_lens_addr.lstrip('0')
  addrs.append(context_lens_addr)
  
  return addrs


def process_paged_attention_v2_kernel(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 2]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  exp_sums_addr = ''.join(reversed_numbers)
  exp_sums_addr = "0x" + exp_sums_addr.lstrip('0')
  addrs.append(exp_sums_addr)
  
  addr_line = all_lines[line_no + 5]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  max_logits_addr = ''.join(reversed_numbers)
  max_logits_addr = "0x" + max_logits_addr.lstrip('0')
  addrs.append(max_logits_addr)
  
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  tmp_out_addr = ''.join(reversed_numbers)
  tmp_out_addr = "0x" + tmp_out_addr.lstrip('0')
  addrs.append(tmp_out_addr)
  
  addr_line = all_lines[line_no + 11]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  q_addr = ''.join(reversed_numbers)
  q_addr = "0x" + q_addr.lstrip('0')
  addrs.append(q_addr)
  
  addr_line = all_lines[line_no + 14]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  k_cache_addr = ''.join(reversed_numbers)
  k_cache_addr = "0x" + k_cache_addr.lstrip('0')
  addrs.append(k_cache_addr)

  addr_line = all_lines[line_no + 17]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  v_cache_addr = ''.join(reversed_numbers)
  v_cache_addr = "0x" + v_cache_addr.lstrip('0')
  addrs.append(v_cache_addr)
  
  addr_line = all_lines[line_no + 26]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  block_tables_addr = ''.join(reversed_numbers)
  block_tables_addr = "0x" + block_tables_addr.lstrip('0')
  addrs.append(block_tables_addr)
  
  addr_line = all_lines[line_no + 29]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  context_lens_addr = ''.join(reversed_numbers)
  context_lens_addr = "0x" + context_lens_addr.lstrip('0')
  addrs.append(context_lens_addr)
  
  addr_line = all_lines[line_no + 35]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  context_lens_addr = ''.join(reversed_numbers)
  context_lens_addr = "0x" + context_lens_addr.lstrip('0')
  addrs.append(context_lens_addr)
  
  return addrs

def process_paged_attention_v2_reduce_kernel(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 2]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  out_addr = ''.join(reversed_numbers)
  out_addr = "0x" + out_addr.lstrip('0')
  addrs.append(out_addr)
  
  addr_line = all_lines[line_no + 5]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  exp_sums_addr = ''.join(reversed_numbers)
  exp_sums_addr = "0x" + exp_sums_addr.lstrip('0')
  addrs.append(exp_sums_addr)
  
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  max_logits_addr = ''.join(reversed_numbers)
  max_logits_addr = "0x" + max_logits_addr.lstrip('0')
  addrs.append(max_logits_addr)
  
  addr_line = all_lines[line_no + 11]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  tmp_out_addr = ''.join(reversed_numbers)
  tmp_out_addr = "0x" + tmp_out_addr.lstrip('0')
  addrs.append(tmp_out_addr)
  
  addr_line = all_lines[line_no + 14]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  context_lens_addr = ''.join(reversed_numbers)
  context_lens_addr = "0x" + context_lens_addr.lstrip('0')
  addrs.append(context_lens_addr)
  
  return addrs

def process_cublasSplitKreduce_kernel(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 11]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 14]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  return addrs

def process_fused_add_rms_norm_kernel(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 2]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  input_addr = ''.join(reversed_numbers)
  input_addr = "0x" + input_addr.lstrip('0')
  addrs.append(input_addr)
  
  addr_line = all_lines[line_no + 5]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  residual_addr = ''.join(reversed_numbers)
  residual_addr = "0x" + residual_addr.lstrip('0')
  addrs.append(residual_addr)
  
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  weight_addr = ''.join(reversed_numbers)
  weight_addr = "0x" + weight_addr.lstrip('0')
  addrs.append(weight_addr)
  
  return addrs


def get_base_addr(addr, offset):
  integer_value = int(addr, 16) - offset
  return str(hex(integer_value))


def process_reshape_and_cache_kernel(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 2]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  key_addr = ''.join(reversed_numbers)
  key_addr = "0x" + key_addr.lstrip('0')
  key_addr = get_base_addr(key_addr, key_offset)
  addrs.append(key_addr)
  
  addr_line = all_lines[line_no + 5]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  value_addr = ''.join(reversed_numbers)
  value_addr = "0x" + value_addr.lstrip('0')
  value_addr = get_base_addr(value_addr, value_offset)
  addrs.append(value_addr)
  
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  key_cache_addr = ''.join(reversed_numbers)
  key_cache_addr = "0x" + key_cache_addr.lstrip('0')
  addrs.append(key_cache_addr)
  
  addr_line = all_lines[line_no + 11]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  value_cache_addr = ''.join(reversed_numbers)
  value_cache_addr = "0x" + value_cache_addr.lstrip('0')
  addrs.append(value_cache_addr)
  
  addr_line = all_lines[line_no + 14]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  slot_mapping_addr = ''.join(reversed_numbers)
  slot_mapping_addr = "0x" + slot_mapping_addr.lstrip('0')
  addrs.append(slot_mapping_addr)
  
  return addrs


def process_rotary_embedding_kernel(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 2]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  positions_addr = ''.join(reversed_numbers)
  positions_addr = "0x" + positions_addr.lstrip('0')
  addrs.append(positions_addr)
  
  addr_line = all_lines[line_no + 5]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  query_addr = ''.join(reversed_numbers)
  query_addr = "0x" + query_addr.lstrip('0')
  addrs.append(query_addr)
  
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  key_addr = ''.join(reversed_numbers)
  key_addr = "0x" + key_addr.lstrip('0')
  key_addr = get_base_addr(key_addr, key_offset)
  addrs.append(key_addr)
  
  addr_line = all_lines[line_no + 11]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  cos_sin_cache_addr = ''.join(reversed_numbers)
  cos_sin_cache_addr = "0x" + cos_sin_cache_addr.lstrip('0')
  addrs.append(cos_sin_cache_addr)
  
  return addrs
  

def process_silu_and_mul_kernel(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 2]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  out_addr = ''.join(reversed_numbers)
  out_addr = "0x" + out_addr.lstrip('0')
  addrs.append(out_addr)
  
  addr_line = all_lines[line_no + 5]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  input_addr = ''.join(reversed_numbers)
  input_addr = "0x" + input_addr.lstrip('0')
  addrs.append(input_addr)
  
  return addrs


def process_rms_norm_kernel(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 2]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  out_addr = ''.join(reversed_numbers)
  out_addr = "0x" + out_addr.lstrip('0')
  addrs.append(out_addr)
  
  addr_line = all_lines[line_no + 5]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  input_addr = ''.join(reversed_numbers)
  input_addr = "0x" + input_addr.lstrip('0')
  addrs.append(input_addr)
  
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  weight_addr = ''.join(reversed_numbers)
  weight_addr = "0x" + weight_addr.lstrip('0')
  addrs.append(weight_addr)
  
  return addrs

def process_vectorized_elementwise_kernel_onself(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[-8:][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  return addrs

def process_vectorized_elementwise_kernel_add(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[-8:][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 9]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  return addrs

def process_layer_norm_kernel(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 11]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 14]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 17]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 20]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 23]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  return addrs


def process_vectorized_elementwise_kernel_gelu(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[-8:][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  return addrs

def process_cutlass_80_tensorop(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 23]
  numbers = addr_line.split()
  reversed_numbers = numbers[-8:][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 24]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 24]
  numbers = addr_line.split()
  reversed_numbers = numbers[-8:][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 25]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  return addrs


def process_cutlass_80_tensorop_360(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 19]
  numbers = addr_line.split()
  reversed_numbers = numbers[-8:][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 20]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 20]
  numbers = addr_line.split()
  reversed_numbers = numbers[-8:][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 21]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  return addrs

def loadGraphKernelNodeFuncParams_cublasGemv(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 2]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 3]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 4]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 5]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  return addrs

def loadGraphKernelNodeFuncParams_cublasGemv_152(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 2]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 3]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 4]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 5]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 8]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  return addrs


def loadGraphKernelNodeFuncParams_cublasGemv_152_2(all_lines, line_no):
  addrs = []
  addr_line = all_lines[line_no + 2]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 3]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  addr_line = all_lines[line_no + 5]
  numbers = addr_line.split()
  reversed_numbers = numbers[:8][::-1]
  addr = ''.join(reversed_numbers)
  addr = "0x" + addr.lstrip('0')
  addrs.append(addr)
  
  return addrs

def filter_all_malloc_free_lines(line, all_malloc_free_lines):
  malloc_match = re.search(malloc_pattern, line)
  free_match = re.search(free_pattern, line)
  if malloc_match or free_match:
    all_malloc_free_lines.append(line)
  if line.find("=============== capture_begin ===============") != -1:
    all_malloc_free_lines.append(line)
  if line.find("=============== capture_end ===============") != -1:
    all_malloc_free_lines.append(line)


def process_malloc_free_line(should_write_to_file, line, output_file, malloc_idx, address_to_malloc_idx_map, malloc_address):
  match = re.search(malloc_pattern, line)
  if match:
      # 提取地址和大小
      address = match.group(1)
      size = match.group(2)
      address_to_malloc_idx_map[address] = malloc_idx[0]
      malloc_idx[0] = malloc_idx[0] + 1
      malloc_address.append(address)
      if should_write_to_file:
        output_file.write(f"m {address_to_malloc_idx_map[address]} {size}\n")
  else:
      match = re.search(free_pattern, line)
      if match:
        # 提取地址
        address = match.group(1)
        idx = address_to_malloc_idx_map[address]
        
        if should_write_to_file:
          output_file.write(f"f {idx}\n")
      else:
        if line.find("=============== capture_begin ===============") != -1:
          if should_write_to_file:
            output_file.write(line)
        elif line.find("=============== capture_end ===============") != -1:
          if should_write_to_file:
            output_file.write(line)
        else:
          print("No match found: ", line)
          exit(-1)
          
def filter_all_nodes_lines(all_lines, all_nodes_lines):
  idx = 0
  begin_nodes = False
  while idx < len(all_lines):
    line = all_lines[idx]
    if line.find("begin_nodes") != -1:
      begin_nodes = True
    
    if begin_nodes:
      all_nodes_lines.append(line)
    
    if line.find("end_nodes") != -1:
      break
    
    idx = idx + 1
    
def filter_dependency_nodes_lines(all_lines, dependency_nodes_lines):
  idx = 0
  begin_nodes = False
  dependency_pattern = r"dependencies from_idx: (\d+), to_idx: (\d+)"
  
  while idx < len(all_lines):
    line = all_lines[idx]
    
    match = re.search(dependency_pattern, line)
    if match:
      dependency_nodes_lines.append(line)
    
    idx = idx + 1
    
def process_nodes_line(all_nodes_lines, ampere_nodes, memset_nodes, other_nodes, cublasssplitK_nodes, node_output_file):
  line_no = 0
  layer_no = 0
  one_layer_nodes = []
  one_layer_nodes_idx = 0
  should_write_to_file = True
  while line_no < len(all_nodes_lines):
    line = all_nodes_lines[line_no]
    
    if line.find("node type") != -1:
      should_write_to_file = True
      
      node_type = get_node_type(line)
      node_idx = get_node_idx(line)
      
      if node_type == 2:
        addrs = process_memset_kernel(all_nodes_lines, line_no)
        node = Node(node_idx, "memset")
        node.addrs = addrs
        memset_nodes.append(node)
        
        if layer_no == 1:
          one_layer_nodes.append(node)

        if layer_no > 1:
          layer_one_node = one_layer_nodes[one_layer_nodes_idx]
          one_layer_nodes_idx = (one_layer_nodes_idx + 1) % len(one_layer_nodes)
          
          if layer_one_node.name != "memset":
            print("layer_one_node.name: ", layer_one_node.name, " not: ", "memset")
            exit(-1)
          else:
            node_output_file.write(line)
            node_output_file.write("============================\n")
            should_write_to_file = False
      
      
    if line.find("kernel_func.name") != -1:
      name = get_node_func_name(line)
      node = Node(node_idx, name)
      
      if layer_no == 1:
        one_layer_nodes.append(node)
        
      if layer_no > 1:
        layer_one_node = one_layer_nodes[one_layer_nodes_idx]
        one_layer_nodes_idx = (one_layer_nodes_idx + 1) % len(one_layer_nodes)
        
        if layer_one_node.name != name:
            print("layer_one_node.name: ", layer_one_node.name, " not: ", name)
            exit(-1)
        else:
            node_output_file.write(line)
            node_output_file.write("============================\n")
            should_write_to_file = False
            
      if line.find("rotary_embedding_kernel") != -1:
        layer_no = layer_no + 1
        
      for i in range(12):
        if should_write_to_file:
          node_output_file.write(line)
        line_no = line_no + 1
        line = all_nodes_lines[line_no]
        
      
      if name.find("ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_relu_f2f_stages_64x5_tn") != -1 or name.find("ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_relu_f2f_stages_64x5_tn") != -1 or name.find("ampere_bf16_s16816gemm_bf16_64x64_ldg8_relu_f2f_stages_64x5_tn") != -1 or name.find("ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_32x6_tn") != -1 or name.find("ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x4_tn") != -1 or name.find("ampere_bf16_s16816gemm_bf16_256x64_ldg8_relu_f2f_stages_64x3_tn") != -1 or name.find("ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_64x3_tn") != -1 or name.find("ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_64x3_tn") != -1 or name.find("ampere_bf16_s16816gemm_bf16_128x256_ldg8_relu_f2f_stages_64x3_tn") != -1 or name.find("ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_relu_f2f_stages_64x6_tn") != -1:
        addrs = process_ampere_kernel_208(all_nodes_lines, line_no)
        node.addrs = addrs
        ampere_nodes.append(node)
      elif name.find("ampere") != -1:
        addrs = process_ampere_kernel(all_nodes_lines, line_no)
        node.addrs = addrs
        ampere_nodes.append(node)
      elif name.find("_ZN7cutlass7Kernel2I66cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x6_tn_align8EEvNT_6ParamsE") != -1 or name.find("_ZN7cutlass7Kernel2I65cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x6_tn_align8EEvNT_6ParamsE") != -1 or name.find("_ZN7cutlass7Kernel2I66cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_tn_align8EEvNT_6ParamsE") != -1 or name.find("_ZN7cutlass7Kernel2I67cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x6_tn_align8EEvNT_6ParamsE") != -1:
        addrs = process_cutlass_80_tensorop(all_nodes_lines, line_no)
        node.addrs = addrs
        ampere_nodes.append(node)
      elif name.find("_ZN7cutlass7Kernel2I56cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8EEvNT_6ParamsE") != -1 or name.find("_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8EEvNT_6ParamsE") != -1 or name.find("_ZN7cutlass7Kernel2I63cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_128x2_tn_align8EEvNT_6ParamsE") != -1 or name.find("_ZN7cutlass7Kernel2I57cutlass_80_tensorop_s16816gemm_bf16_64x128_64x4_tn_align8EEvNT_6ParamsE") != -1 or name.find("_ZN7cutlass7Kernel2I68cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8EEvNT_6ParamsE") != -1:
        addrs = process_cutlass_80_tensorop_360(all_nodes_lines, line_no)
        node.addrs = addrs
        ampere_nodes.append(node)
      elif name.find("_ZN8internal5gemvx6kernelIii13__nv_bfloat16S2_S2_fLb0ELb1ELb1ELb0ELi7ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS2_ES6_S4_IS2_EfEEENSt9enable_ifIXntT5_EvE4typeET11_") != -1:
        addrs = loadGraphKernelNodeFuncParams_cublasGemv(all_nodes_lines, line_no)
        node.addrs = addrs
        ampere_nodes.append(node)
      elif name.find("_Z17gemv2T_kernel_valIii13__nv_bfloat16S0_S0_fLi128ELi16ELi4ELi4ELb0ELb1E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS0_ES4_S2_IS0_EfEEvT11_T4_S8_") != -1:
        addrs = loadGraphKernelNodeFuncParams_cublasGemv_152(all_nodes_lines, line_no)
        node.addrs = addrs
        ampere_nodes.append(node)
      elif name.find("_Z17gemv2T_kernel_valIii13__nv_bfloat16fffLi128ELi16ELi4ELi4ELb0ELb0E18cublasGemvParamsExIi30cublasGemvTensorStridedBatchedIKS0_ES2_IKfES2_IfEfEEvT11_T4_SA_") != -1:
        addrs = loadGraphKernelNodeFuncParams_cublasGemv_152_2(all_nodes_lines, line_no)
        node.addrs = addrs
        ampere_nodes.append(node)
      elif name.find("sm80_xmma_gemm") != -1:
        addrs = process_sm80_kernel(all_nodes_lines, line_no)
        node.addrs = addrs
        ampere_nodes.append(node)
      elif name.find("cublasLt19splitKreduce_kernel") != -1:
        addrs = process_cublasSplitKreduce_kernel(all_nodes_lines, line_no)
        node.addrs = addrs
        ampere_nodes.append(node)
      elif name.find("paged_attention_v1_kernel") != -1:
        addrs = process_paged_attention_v1_kernel(all_nodes_lines, line_no)
        node.addrs = addrs
        other_nodes.append(node)
      elif name.find("paged_attention_v2_kernel") != -1:
        addrs = process_paged_attention_v2_kernel(all_nodes_lines, line_no)
        node.addrs = addrs
        other_nodes.append(node)
      elif name.find("paged_attention_v2_reduce_kernel") != -1:
        addrs = process_paged_attention_v2_reduce_kernel(all_nodes_lines, line_no)
        node.addrs = addrs
        other_nodes.append(node)
      elif name.find("_ZN4vllm25fused_add_rms_norm_kernelIN3c104HalfEEEvPT_S4_PKS3_fii") != -1 or name.find("_ZN4vllm25fused_add_rms_norm_kernelIN3c108BFloat16EEEvPT_S4_PKS3_fii") != -1:
        addrs = process_fused_add_rms_norm_kernel(all_nodes_lines, line_no)
        node.addrs = addrs
        other_nodes.append(node)
      elif name.find("_ZN4vllm24reshape_and_cache_kernelIttLb0EEEvPKT_S3_PT0_S5_PKliiiiii") != -1 or name.find("_ZN4vllm24reshape_and_cache_kernelI13__nv_bfloat16S1_Lb0EEEvPKT_S4_PT0_S6_PKliiiiii") != -1:
        addrs = process_reshape_and_cache_kernel(all_nodes_lines, line_no)
        node.addrs = addrs
        other_nodes.append(node)
      elif name.find("_ZN4vllm23rotary_embedding_kernelIN3c104HalfELb1EEEvPKlPT_S6_PKS5_illiii") != -1 or name.find("_ZN4vllm23rotary_embedding_kernelIN3c108BFloat16ELb1EEEvPKlPT_S6_PKS5_illiii") != -1:
        addrs = process_rotary_embedding_kernel(all_nodes_lines, line_no)
        node.addrs = addrs
        other_nodes.append(node)
      elif name.find("_ZN4vllm19silu_and_mul_kernelIN3c104HalfEEEvPT_PKS3_i") != -1 or name.find("_ZN4vllm19silu_and_mul_kernelIN3c108BFloat16EEEvPT_PKS3_i") != -1:
        addrs = process_silu_and_mul_kernel(all_nodes_lines, line_no)
        node.addrs = addrs
        other_nodes.append(node)
      elif name.find("_ZN4vllm15rms_norm_kernelIN3c104HalfEEEvPT_PKS3_S6_fii") != -1 or name.find("_ZN4vllm15rms_norm_kernelIN3c108BFloat16EEEvPT_PKS3_S6_fii") != -1:
        addrs = process_rms_norm_kernel(all_nodes_lines, line_no)
        node.addrs = addrs
        other_nodes.append(node)
      elif name.find("_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_21CUDAFunctorOnSelf_addIlEENS_6detail5ArrayIPcLi2EEEEEviT0_T1_") != -1 or name.find("_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_18GeluCUDAKernelImplERNS_18TensorIteratorBaseENS0_8GeluTypeEENKUlvE0_clEvENKUlvE2_clEvEUlN3c108BFloat16EE_NS_6detail5ArrayIPcLi2EEEEEviT0_T1_") != -1:
        addrs = process_vectorized_elementwise_kernel_onself(all_nodes_lines, line_no)
        node.addrs = addrs
        other_nodes.append(node)
      elif name.find("_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_15CUDAFunctor_addIN3c104HalfEEENS_6detail5ArrayIPcLi3EEEEEviT0_T1_") != -1 or name.find("_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_15CUDAFunctor_addIN3c108BFloat16EEENS_6detail5ArrayIPcLi3EEEEEviT0_T1_") != -1:
        addrs = process_vectorized_elementwise_kernel_add(all_nodes_lines, line_no)
        node.addrs = addrs
        other_nodes.append(node)
      elif name.find("_ZN2at6native53_GLOBAL__N__1aa0d23f_20_layer_norm_kernel_cu_9c5ada8a28vectorized_layer_norm_kernelIN3c104HalfEfEEviT0_PKT_S8_S8_PS5_S9_PS6_") != -1 or name.find("_ZN2at6native53_GLOBAL__N__1aa0d23f_20_layer_norm_kernel_cu_9c5ada8a28vectorized_layer_norm_kernelIN3c108BFloat16EfEEviT0_PKT_S8_S8_PS5_S9_PS6_") != -1 or name.find("_ZN2at6native53_GLOBAL__N__7ae05bd5_20_layer_norm_kernel_cu_9c5ada8a28vectorized_layer_norm_kernelIN3c108BFloat16EfEEviT0_PKT_S8_S8_PS5_S9_PS6_") != -1:
        addrs = process_layer_norm_kernel(all_nodes_lines, line_no)
        node.addrs = addrs
        other_nodes.append(node)
      elif name.find("_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_18GeluCUDAKernelImplERNS_18TensorIteratorBaseENS0_8GeluTypeEENKUlvE0_clEvENKUlvE1_clEvEUlN3c104HalfEE_NS_6detail5ArrayIPcLi2EEEEEviT0_T1_") != -1:
        addrs = process_vectorized_elementwise_kernel_gelu(all_nodes_lines, line_no)
        node.addrs = addrs
        other_nodes.append(node)
    
    if should_write_to_file:
      node_output_file.write(line)
    line_no = line_no + 1
    

def process_dependency_line(all_dependency_lines, dependency_output_file):
  line_no = 0
  dependency_pattern = r"dependencies from_idx: (\d+), to_idx: (\d+)"
  
  while line_no < len(all_dependency_lines):
    line = all_dependency_lines[line_no]
    
    match = re.search(dependency_pattern, line)
    if match:
        from_idx = match.group(1)
        to_idx = match.group(2)
      
        dependency_output_file.write(f"{from_idx} {to_idx}\n")

    line_no = line_no + 1
  

def process_node_to_malloc_idx(all_lines, batch_idx, ampere_to_malloc_idx, other_nodes_to_malloc_idx, memset_to_malloc_idx):
  capture_begin = False
  line_no = 0
  malloc_num = 0
  while True:
    # within capture_begin and capture_end
    line = all_lines[line_no]
    line_no = line_no + 1
    malloc_match = re.search(malloc_pattern, line)
    if malloc_match:
      malloc_num = malloc_num + 1
      
      
    if line.find("=============== capture_begin ===============") != -1:
      if batch_idx == 0:
        capture_begin = True
      else:
        batch_idx = batch_idx - 1
      continue
    if line.find("=============== capture_end ===============") != -1:
      if capture_begin:
        break
    
    if capture_begin:
      if line == "cublass \n":
        ampere_to_malloc_idx.append(malloc_num - 1)
      elif line.find("cudaLaunchKernel hooked") != -1:
        if line.find("paged_attention_v1_kernel") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
        elif line.find("paged_attention_v2_kernel") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
        elif line.find("paged_attention_v2_reduce_kernel") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
        elif line.find("_ZN4vllm15rms_norm_kernelIN3c104HalfEEEvPT_PKS3_S6_fii") != -1 or line.find("_ZN4vllm15rms_norm_kernelIN3c108BFloat16EEEvPT_PKS3_S6_fii") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
        elif line.find("_ZN4vllm24reshape_and_cache_kernelIttLb0EEEvPKT_S3_PT0_S5_PKliiiiii") != -1 or line.find("_ZN4vllm24reshape_and_cache_kernelI13__nv_bfloat16S1_Lb0EEEvPKT_S4_PT0_S6_PKliiiiii") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
        elif line.find("_ZN4vllm23rotary_embedding_kernelIN3c104HalfELb1EEEvPKlPT_S6_PKS5_illiii") != -1 or line.find("_ZN4vllm23rotary_embedding_kernelIN3c108BFloat16ELb1EEEvPKlPT_S6_PKS5_illiii") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
        elif line.find("_ZN4vllm19silu_and_mul_kernelIN3c104HalfEEEvPT_PKS3_i") != -1 or line.find("_ZN4vllm19silu_and_mul_kernelIN3c108BFloat16EEEvPT_PKS3_i") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
        elif line.find("_ZN4vllm25fused_add_rms_norm_kernelIN3c104HalfEEEvPT_S4_PKS3_fii") != -1 or line.find("_ZN4vllm25fused_add_rms_norm_kernelIN3c108BFloat16EEEvPT_S4_PKS3_fii") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
        elif line.find("_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_21CUDAFunctorOnSelf_addIlEENS_6detail5ArrayIPcLi2EEEEEviT0_T1_") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
        elif line.find("_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_15CUDAFunctor_addIN3c104HalfEEENS_6detail5ArrayIPcLi3EEEEEviT0_T1_") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
        elif line.find("layer_norm_kernel") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
        elif line.find("_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_49_GLOBAL__N__77751aff_16_TensorCompare_cu_58b5c4e919launch_clamp_scalarERNS_18TensorIteratorBaseEN3c106ScalarES6_NS0_6detail11ClampLimitsEENKUlvE_clEvENKUlvE6_clEvEUlNS5_4HalfEE_NS_6detail5ArrayIPcLi2EEEEEviT0_T1_") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
        elif line.find("_ZN2at6native53_GLOBAL__N__1aa0d23f_20_layer_norm_kernel_cu_9c5ada8a28vectorized_layer_norm_kernelIN3c104HalfEfEEviT0_PKT_S8_S8_PS5_S9_PS6_") != -1 or line.find("_ZN2at6native53_GLOBAL__N__1aa0d23f_20_layer_norm_kernel_cu_9c5ada8a28vectorized_layer_norm_kernelIN3c108BFloat16EfEEviT0_PKT_S8_S8_PS5_S9_PS6_") != -1 or line.find("_ZN2at6native53_GLOBAL__N__7ae05bd5_20_layer_norm_kernel_cu_9c5ada8a28vectorized_layer_norm_kernelIN3c108BFloat16EfEEviT0_PKT_S8_S8_PS5_S9_PS6_") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
        elif line.find("_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_18GeluCUDAKernelImplERNS_18TensorIteratorBaseENS0_8GeluTypeEENKUlvE0_clEvENKUlvE1_clEvEUlN3c104HalfEE_NS_6detail5ArrayIPcLi2EEEEEviT0_T1_") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
        elif line.find("_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_15CUDAFunctor_addIN3c108BFloat16EEENS_6detail5ArrayIPcLi3EEEEEviT0_T1_") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
        elif line.find("_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_18GeluCUDAKernelImplERNS_18TensorIteratorBaseENS0_8GeluTypeEENKUlvE0_clEvENKUlvE2_clEvEUlN3c108BFloat16EE_NS_6detail5ArrayIPcLi2EEEEEviT0_T1_") != -1:
          other_nodes_to_malloc_idx.append(malloc_num - 1)
  
  
def find_addr_corresponding_malloc_idx(addr, last_malloc_idx, malloc_address, address_to_malloc_idx_map):
  if addr not in address_to_malloc_idx_map:
    return -1
  while True:
    if malloc_address[last_malloc_idx] == addr:
      return last_malloc_idx
    last_malloc_idx = last_malloc_idx - 1
    if last_malloc_idx < 0:
      return -1
    
def fill_cublasSplitKreduce_to_malloc_idx(ampere_nodes, cublasssplitK_nodes, ampere_to_malloc_idx, cublasssplitK_to_malloc_idx):
  for node in cublasssplitK_nodes:
    cublasssplitK_node_idx = node.idx
    
    previous_ampere_node = ampere_nodes[0]
    previous_ampere_node_idx = 0
    found = False
    for i in range(len(ampere_nodes)):
      if ampere_nodes[i].idx > cublasssplitK_node_idx:
        break
      previous_ampere_node = ampere_nodes[i]
      previous_ampere_node_idx = i
      
      
    cublasssplitK_to_malloc_idx.append(ampere_to_malloc_idx[previous_ampere_node_idx])
    
def fill_ampere_to_malloc_idx(ampere_to_malloc_idx, ampere_nodes, other_nodes_to_malloc_idx, other_nodes):
  for ampere_node in ampere_nodes:
    ampere_node_idx = ampere_node.idx
    found = False
    for node in other_nodes:
      if node.idx > ampere_node_idx:
        ampere_to_malloc_idx.append(other_nodes_to_malloc_idx[other_nodes.index(node)])
        found = True
        break
    if not found:
      print("fill_ampere_to_malloc_idx not found")
      exit(-1)

def fill_memset_to_malloc_idx(memset_to_malloc_idx, memset_nodes, other_nodes_to_malloc_idx, other_nodes):
  for memset_node in memset_nodes:
    memset_node_idx = memset_node.idx
    found = False
    for node in other_nodes:
      if node.idx > memset_node_idx:
        memset_to_malloc_idx.append(other_nodes_to_malloc_idx[other_nodes.index(node)])
        found = True
        break
    if not found:
      print("fill_memset_to_malloc_idx not found")
      exit(-1)

def process_single_batch(all_lines, batch_idx, all_malloc_free_lines, all_nodes_lines, all_dependency_lines):
  address_to_malloc_idx_map = {}
  malloc_address = []

  ampere_nodes = []
  ampere_to_malloc_idx = []

  cublasssplitK_nodes = []
  cublasssplitK_to_malloc_idx = []

  memset_nodes = []
  memset_to_malloc_idx = []

  other_nodes = []
  other_nodes_to_malloc_idx = []
  
  malloc_idx = [0]

  with open(mf_output_file_path, "w") as mf_output_file:
    should_write_to_file = False
    last_capture_idx = len(all_malloc_free_lines) - (all_malloc_free_lines[::-1].index("=============== capture_begin ===============\n") + 1)
    for i in range(len(all_malloc_free_lines)):
      line = all_malloc_free_lines[i]
      if i >= last_capture_idx and not should_write_to_file:
        mf_output_file.write(str(malloc_idx[0]))
        mf_output_file.write("\n")
        should_write_to_file = True
      process_malloc_free_line(should_write_to_file, line, mf_output_file, malloc_idx, address_to_malloc_idx_map, malloc_address)

  with open(node_output_file_path, "w") as node_output_file:
    process_nodes_line(all_nodes_lines, ampere_nodes, memset_nodes, other_nodes, cublasssplitK_nodes, node_output_file)

  with open(dependency_output_file_path, "w") as dependency_output_file:
    process_dependency_line(all_dependency_lines, dependency_output_file)

  # ======================= get node to malloc idx ========================
  # within capture_begin and capture_end
  process_node_to_malloc_idx(all_lines, batch_idx, ampere_to_malloc_idx, other_nodes_to_malloc_idx, memset_to_malloc_idx)
  
  
  memset_idx = 0
  if (len(memset_to_malloc_idx) != len(memset_nodes)):
    print(f"memset_to_malloc_idx: {len(memset_to_malloc_idx)}, memset_nodes: {len(memset_nodes)}")
    print("re-fill memset_to_malloc_idx")
    memset_to_malloc_idx = []
    fill_memset_to_malloc_idx(memset_to_malloc_idx, memset_nodes, other_nodes_to_malloc_idx, other_nodes)
  while True and len(memset_nodes) > 0:
    memset_node = memset_nodes[memset_idx]
    all_addr_corresponding_malloc_idx = []
    for addr in memset_node.addrs:
      all_addr_corresponding_malloc_idx.append(find_addr_corresponding_malloc_idx(addr, memset_to_malloc_idx[memset_idx], malloc_address, address_to_malloc_idx_map))
    memset_node.all_addr_corresponding_malloc_idx = all_addr_corresponding_malloc_idx
    memset_idx = memset_idx + 1
    if len(memset_to_malloc_idx) == memset_idx:
      break

      
  # the first six kernels are the triggerd matmuls
  # already remove by pytorch output
  # ampere_to_malloc_idx = ampere_to_malloc_idx[6:]
  cublass_idx = 0
  if (len(ampere_to_malloc_idx) != len(ampere_nodes)):
    print(f"ampere_to_malloc_idx: {len(ampere_to_malloc_idx)}, ampere_nodes: {len(ampere_nodes)}")
    # print("shit, please fix")
    # while len(ampere_to_malloc_idx) < len(ampere_nodes):
    #   ampere_to_malloc_idx.append(ampere_to_malloc_idx[-1])
    # exit(-1)
    print("re-fill ampere_to_malloc_idx")
    ampere_to_malloc_idx = []
    fill_ampere_to_malloc_idx(ampere_to_malloc_idx, ampere_nodes, other_nodes_to_malloc_idx, other_nodes)
  while True and len(ampere_nodes) > 0:
    ampere_node = ampere_nodes[cublass_idx]
    all_addr_corresponding_malloc_idx = []
    for addr in ampere_node.addrs:
      all_addr_corresponding_malloc_idx.append(find_addr_corresponding_malloc_idx(addr, ampere_to_malloc_idx[cublass_idx], malloc_address, address_to_malloc_idx_map))
    ampere_node.all_addr_corresponding_malloc_idx = all_addr_corresponding_malloc_idx
    cublass_idx = cublass_idx + 1
    if len(ampere_to_malloc_idx) == cublass_idx:
      break

  other_nodes_idx = 0
  if (len(other_nodes_to_malloc_idx) != len(other_nodes)):
    print("shit (len(other_nodes_to_malloc_idx) != len(other_nodes))")
    exit(-1)
  while True and len(other_nodes) > 0:
    other_node = other_nodes[other_nodes_idx]
    all_addr_corresponding_malloc_idx = []
    for addr in other_node.addrs:
      all_addr_corresponding_malloc_idx.append(find_addr_corresponding_malloc_idx(addr, other_nodes_to_malloc_idx[other_nodes_idx], malloc_address, address_to_malloc_idx_map))
    other_node.all_addr_corresponding_malloc_idx = all_addr_corresponding_malloc_idx
    other_nodes_idx = other_nodes_idx + 1
    if len(other_nodes_to_malloc_idx) == other_nodes_idx:
      break


  with open(func_param_output_file_path, "w") as func_param_output_file:
    for node in ampere_nodes:
      func_param_output_file.write(f"node idx: {node.idx}\n")
      func_param_output_file.write(f"kernel_func.name mangled = {node.name}\n")
      all_addr_corresponding_malloc_idx_str = ",".join(str(x) for x in node.all_addr_corresponding_malloc_idx) + ","
      func_param_output_file.write(f"addr_corresponding_malloc_idx: {all_addr_corresponding_malloc_idx_str}\n")
      func_param_output_file.write("\n")
      
    for node in other_nodes:
      func_param_output_file.write(f"node idx: {node.idx}\n")
      func_param_output_file.write(f"kernel_func.name mangled = {node.name}\n")
      all_addr_corresponding_malloc_idx_str = ",".join(str(x) for x in node.all_addr_corresponding_malloc_idx) + ","
      func_param_output_file.write(f"addr_corresponding_malloc_idx: {all_addr_corresponding_malloc_idx_str}\n")
      func_param_output_file.write("\n")
      
    for node in memset_nodes:
      func_param_output_file.write(f"node idx: {node.idx}\n")
      func_param_output_file.write(f"kernel_func.name mangled = {node.name}\n")
      all_addr_corresponding_malloc_idx_str = ",".join(str(x) for x in node.all_addr_corresponding_malloc_idx) + ","
      func_param_output_file.write(f"addr_corresponding_malloc_idx: {all_addr_corresponding_malloc_idx_str}\n")
      func_param_output_file.write("\n")
      
    for node in cublasssplitK_nodes:
      func_param_output_file.write(f"node idx: {node.idx}\n")
      func_param_output_file.write(f"kernel_func.name mangled = {node.name}\n")
      all_addr_corresponding_malloc_idx_str = ",".join(str(x) for x in node.all_addr_corresponding_malloc_idx) + ","
      func_param_output_file.write(f"addr_corresponding_malloc_idx: {all_addr_corresponding_malloc_idx_str}\n")
      func_param_output_file.write("\n")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, help='batch size, for file name', required=True)
parser.add_argument('--model', type=str, help='model name', required=True)
args = parser.parse_args()

input_file_path = "/mnt/memfs/data/{}/log_graph_{}".format(args.model, args.batch_size)
mf_output_file_path = "/mnt/memfs/data/{}/log_graph_{}_mf".format(args.model, args.batch_size)
node_output_file_path = "/mnt/memfs/data/{}/log_graph_{}_node".format(args.model, args.batch_size)
func_param_output_file_path = "/mnt/memfs/data/{}/log_graph_{}_func_param".format(args.model, args.batch_size)
dependency_output_file_path = "/mnt/memfs/data/{}/log_graph_{}_dependency".format(args.model, args.batch_size)

if args.model == "Llama-13B":
  key_offset = 10240
  value_offset = 20480
elif args.model == "Llama-7B":
  key_offset = 8192
  value_offset = 8192 + 8192
elif args.model == "Llama-3B":
  key_offset = 8192
  value_offset = 8192 + 8192
elif args.model == "Yi-6B":
  key_offset = 8192
  value_offset = 8192 + 1024
elif args.model == "Yi-9B":
  key_offset = 8192
  value_offset = 8192 + 1024
elif args.model == "Bloom":
  key_offset = 8192
  value_offset = 8192 + 8192
elif args.model == "Qwen-14B":
  key_offset = 10240
  value_offset = 10240 + 10240
elif args.model == "Qwen-7B":
  key_offset = 8192
  value_offset = 8192 + 8192
elif args.model == "Qwen-4B":
  key_offset = 5120
  value_offset = 5120 + 5120
elif args.model == "Qwen-1.8B":
  key_offset = 4096
  value_offset = 4096 + 4096
elif args.model == "Qwen-0.5B":
  key_offset = 2048
  value_offset = 2048 + 2048
elif args.model == "Falcon-1B":
  key_offset = 4096
  value_offset = 4096 + 4096
elif args.model == "Falcon-7B":
  key_offset = 9088
  value_offset = 9088 + 128
else:
  print("model name not supported", args.model)
  exit(-1)

# ======================= get all lines ========================
all_lines = []
with open(input_file_path, "r") as input_file:
    for line in input_file:
      all_lines.append(line)
      
# get lines before capture_begin
lines_before_capture_begin = []
for line in all_lines:
  if (line.find("=============== capture_begin ===============") != -1):
    break
  lines_before_capture_begin.append(line)

# get all lines (all batch sizes) between capture_begin and capture_end
lines_between_capture_all_batch = []
lines_between_capture = []
capture_begin = False
for line in all_lines:
  if (line.find("=============== capture_begin ===============") != -1) and not capture_begin:
    capture_begin = True
    lines_between_capture.append(line)
    continue
  if (line.find("=============== capture_end ===============") != -1):
    lines_between_capture.append(line)
    lines_between_capture_all_batch.append(lines_between_capture)
    lines_between_capture = []
    continue
  if capture_begin:
    lines_between_capture.append(line)
    continue

# keep the same in model_runner.py
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [8 * i for i in range(1, 33)]
reversed_batch_size_capture_list = _BATCH_SIZES_TO_CAPTURE[::-1]
batch_idx = 0

# process a single batch one-by-one
lines_before_batch_ends = lines_before_capture_begin
for lines_between_capture_single_batch in lines_between_capture_all_batch:
  if args.batch_size == 0:
    bs = reversed_batch_size_capture_list[batch_idx]
    batch_idx = batch_idx + 1
  else:
    bs = args.batch_size

  input_file_path = "/mnt/memfs/data/{}/log_graph_{}".format(args.model, bs)
  mf_output_file_path = "/mnt/memfs/data/{}/log_graph_{}_mf".format(args.model, bs)
  node_output_file_path = "/mnt/memfs/data/{}/log_graph_{}_node".format(args.model, bs)
  func_param_output_file_path = "/mnt/memfs/data/{}/log_graph_{}_func_param".format(args.model, bs)
  dependency_output_file_path = "/mnt/memfs/data/{}/log_graph_{}_dependency".format(args.model, bs)
    
  # ======================= get all nodes lines ========================
  # within begin_nodes and end_nodes
  all_nodes_lines = []
  filter_all_nodes_lines(lines_between_capture_single_batch, all_nodes_lines)

  # ======================= get dependencies ========================
  # within begin_nodes and end_nodes
  all_dependency_lines = []
  filter_dependency_nodes_lines(lines_between_capture_single_batch, all_dependency_lines)
  
  # need all lines before this batch ends (capture_end)
  lines_before_batch_ends = lines_before_batch_ends + lines_between_capture_single_batch
  # ======================= all malloc/free lines ========================
  # including all malloc and free before this batch ends
  all_malloc_free_lines = []
  for line in lines_before_batch_ends:
    filter_all_malloc_free_lines(line, all_malloc_free_lines)
  if batch_idx != 0:
    process_single_batch(lines_before_batch_ends, batch_idx - 1, all_malloc_free_lines, all_nodes_lines, all_dependency_lines)
  else:
    process_single_batch(lines_before_batch_ends, 0, all_malloc_free_lines, all_nodes_lines, all_dependency_lines)
