#pragma once
enum io_task_type { READ, WRITE };
void spdk_init(int num_disk);
void spdk_io(int nvme_id, void* cpu_ptr,unsigned long offset,unsigned long size,enum io_task_type task);
void spdk_print_stat();