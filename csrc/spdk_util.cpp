#include "spdk_util.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <spdk/env.h>
#include <spdk/nvme.h>

#include <spdk/event.h>
#include "util.h"

#include <vector>

int node_idx;
struct ns_entry {
    struct spdk_nvme_ctrlr* ctrlr;
    struct spdk_nvme_ns* ns;
    struct spdk_nvme_qpair* qpair[THREAD_NUM];
};
struct ctrlr_entry {
    struct spdk_nvme_ctrlr* ctrlr;
    char name[1024];
};

#define q_depth 16
int nvme_max = -1;
struct ns_entry* ns_entry[4];
struct ctrlr_entry* ctrlr_entry[4];
int nvme_cnt = 0;
int init_nvme_no = -1;
// static int gpu_id;
static struct spdk_nvme_transport_id g_trid = {};
static void register_ns(struct spdk_nvme_ctrlr* ctrlr, struct spdk_nvme_ns* ns)
{
    struct ns_entry* entry;

    if (!spdk_nvme_ns_is_active(ns)) { return; }
    int size = spdk_nvme_ns_get_sector_size(ns);
    printf("sector size %d\n", size);
    if (size != sector_size) {
        printf("ERROR: sector size is not 4096\n");
        exit(1);
    }
    
    entry = (struct ns_entry*)malloc(sizeof(struct ns_entry));
    if (entry == NULL) {
        perror("ns_entry malloc");
        exit(1);
    }

    entry->ctrlr = ctrlr;
    entry->ns = ns;
    ns_entry[init_nvme_no] = entry;
    printf("  Namespace ID: %d size: %juGB\n",
           spdk_nvme_ns_get_id(ns),
           spdk_nvme_ns_get_size(ns) / 1000000000);
    for (int i = 0; i < THREAD_NUM; i++) {
        entry->qpair[i] = spdk_nvme_ctrlr_alloc_io_qpair(entry->ctrlr, NULL, 0);
        if (entry->qpair[i] == NULL) {
            printf("ERROR: spdk_nvme_ctrlr_alloc_io_qpair() failed\n");
            return;
        }
    }
}
static int cnt = 0;
static bool probe_cb(void* cb_ctx,
                     const struct spdk_nvme_transport_id* trid,
                     struct spdk_nvme_ctrlr_opts* opts)
{
    cnt++;
    if(cnt<=nvme_max){
        return true;
    }
    return false;
}
static void attach_cb(void* cb_ctx,
                      const struct spdk_nvme_transport_id* trid,
                      struct spdk_nvme_ctrlr* ctrlr,
                      const struct spdk_nvme_ctrlr_opts* opts)
{
    int nsid;
    struct ctrlr_entry* entry;
    struct spdk_nvme_ns* ns;
    const struct spdk_nvme_ctrlr_data* cdata;

    entry = (struct ctrlr_entry*)malloc(sizeof(struct ctrlr_entry));
    if (entry == NULL) {
        perror("ctrlr_entry malloc");
        exit(1);
    }

    printf("%d Attached to %s\n",node_idx,trid->traddr);

    /*
     * spdk_nvme_ctrlr is the logical abstraction in SPDK for an NVMe
     *  controller.  During initialization, the IDENTIFY data for the
     *  controller is read using an NVMe admin command, and that data
     *  can be retrieved using spdk_nvme_ctrlr_get_data() to get
     *  detailed information on the controller.  Refer to the NVMe
     *  specification for more details on IDENTIFY for NVMe controllers.
     */
    cdata = spdk_nvme_ctrlr_get_data(ctrlr);

    snprintf(entry->name, sizeof(entry->name), "%-20.20s (%-20.20s)", cdata->mn, cdata->sn);

    entry->ctrlr = ctrlr;
    if (strcmp(trid->traddr, "0000:c2:00.0") == 0) {
        init_nvme_no = 0;
    } else if (strcmp(trid->traddr, "0000:63:00.0") == 0) {
        init_nvme_no = 1;
    } else if (strcmp(trid->traddr, "0000:c3:00.0") == 0) {
        init_nvme_no = 2;
    } else if (strcmp(trid->traddr, "0000:c5:00.0") == 0) {
        init_nvme_no = 3;
    } else {
        printf("ERROR: unknown nvme device!\n");
        exit(1);
    }
    printf("init_nvme_no %d\n", init_nvme_no);
    ctrlr_entry[init_nvme_no] = entry;

    /*
     * Each controller has one or more namespaces.  An NVMe namespace is basically
     *  equivalent to a SCSI LUN.  The controller's IDENTIFY data tells us how
     *  many namespaces exist on the controller.  For Intel(R) P3X00 controllers,
     *  it will just be one namespace.
     *
     * Note that in NVMe, namespace IDs start at 1, not 0.
     */
    for (nsid = spdk_nvme_ctrlr_get_first_active_ns(ctrlr); nsid != 0;
         nsid = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, nsid)) {
        ns = spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
        if (ns == NULL) { continue; }
        register_ns(ctrlr, ns);
    }
    fflush(stdout);
    nvme_cnt++;
}
void* hello_start(void* arg){

}

uint64_t nvme_max_offset[128];
void spdk_init(int num_disk)
{
    nvme_max = num_disk;
    struct spdk_env_opts opts;
    spdk_env_opts_init(&opts);
    opts.shm_id = 114514;
    opts.name = "hello_world";
    assert(THREAD_NUM == 4);
    opts.core_mask = "[0,1,2,3,4]";
    opts.mem_size = THREAD_NUM * 1024;
    spdk_nvme_trid_populate_transport(&g_trid, SPDK_NVME_TRANSPORT_PCIE);
    snprintf(g_trid.subnqn, sizeof(g_trid.subnqn), "%s", SPDK_NVMF_DISCOVERY_NQN);
    if (spdk_env_init(&opts) < 0) {
        fprintf(stderr, "Unable to initialize SPDK env\n");
        exit(1);
    }
    int rc = spdk_nvme_probe(&g_trid, NULL, probe_cb, attach_cb, NULL);
    if (rc != 0) {
        fprintf(stderr, "spdk_nvme_probe() failed\n");
        rc = 1;
        exit(1);
    }

    for (int i = 0; i < 128; i++) {
        nvme_max_offset[i] = -1;
    }
}

struct io_task {
    void* buf;
    uint64_t start_offset;
    size_t total_size;
    int total_blocks;
    int finished_blocks;
    size_t sent_blocks;
    enum io_task_type type;
};
size_t total_io_size = 0;
unsigned long total_io_time = 0;
void io_callback(void* task, const struct spdk_nvme_cpl* cpl);
unsigned long cpu_cycle = 0;
uint64_t abs_offset_max = -1;
static inline void do_io(struct io_task* task)
{
    unsigned long start = rdtscll();
    if (task->sent_blocks == task->total_blocks) return;
    assert(task->total_size % sector_size == 0);
    assert(task->start_offset % block_size == 0);
    size_t abs_offset = task->start_offset+ task->sent_blocks * block_size;
    if (task->type == WRITE && abs_offset <= abs_offset_max && abs_offset_max != -1) {
        printf("ERROR: write to the same offset!\n");
        printf("former abs_offset %ld, abs_offset %ld\n", abs_offset_max, abs_offset);
        exit(-1);
    }
    abs_offset_max = abs_offset;
    size_t abs_block_id = abs_offset / block_size;
    size_t abs_lba = abs_offset / block_size / nvme_cnt * (block_size / sector_size) + (abs_offset / block_size % nvme_cnt) / sector_size;
    int nvme_id = abs_block_id % nvme_cnt;
    int cnt = std::min(block_size, task->total_size - task->sent_blocks * block_size) / sector_size;
    // printf("nvme_id %d %s abs_lba %ld cnt %d offset %ld\n", nvme_id, ctrlr_entry[nvme_id]->name, abs_lba, cnt, task->sent_blocks * block_size);
    if (task->type == WRITE && nvme_max_offset[nvme_id] >= abs_lba && nvme_max_offset[nvme_id] != -1) {
        printf("ERROR: write to the same block!\n");
        printf("nvme_id %d former abs_lba %ld, abs_lba %ld\n", nvme_id, nvme_max_offset[nvme_id], abs_lba);
        exit(1);
    }
    nvme_max_offset[nvme_id] = abs_lba;
    struct spdk_nvme_ns* ns = ns_entry[nvme_id]->ns;
    struct spdk_nvme_qpair* qpair = ns_entry[nvme_id]->qpair[thread_idx];
    
    if (task->type == READ) {
        int ret = spdk_nvme_ns_cmd_read(
            ns,
            qpair,
            task->buf + task->sent_blocks * block_size,
            abs_lba,
            std::min(block_size, task->total_size - task->sent_blocks * block_size) / sector_size,
            io_callback,
            task,
            0);

    } else {
        spdk_nvme_ns_cmd_write(
            ns,
            qpair,
            task->buf + task->sent_blocks * block_size,
            abs_lba,
            std::min(block_size, task->total_size - task->sent_blocks * block_size) / sector_size,
            io_callback,
            task,
            0);
    }
    task->sent_blocks++;
    cpu_cycle +=rdtscll()-start;
}
void io_callback(void* task, const struct spdk_nvme_cpl* cpl)
{
    if (spdk_nvme_cpl_is_error(cpl)) {
        printf("read error!\n");
        printf("status is %d\n",cpl->status.sct);
        printf("error string is %s\n",spdk_nvme_cpl_get_status_string(&cpl->status));
        exit(1);
    }
    struct io_task* t = (struct io_task*)task;
    t->finished_blocks++;
    do_io((struct io_task*)task);
    return;
}
void spdk_io(int nvme_id, void* cpu_ptr, unsigned long offset, unsigned long size, enum io_task_type type)
{
    if(nvme_cnt==0){
        printf("wtf, no nvme device!\n");
        fflush(stdout);
        abort();
    }
    // printf("%s %ld bytes from %ld\n", type == READ ? "read" : "write", size, offset);
    unsigned long start = rdtscll();
    // we try lock it and assert no others
    struct io_task task;
    size = round_up(size, sector_size);
    task.type = type;
    task.buf = cpu_ptr;
    task.total_size = size;
    task.total_blocks = round_up_divide(size, block_size);
    task.finished_blocks = 0;
    task.sent_blocks = 0;
    task.start_offset = offset ;
    assert(offset % block_size == 0);
    // printf("start lba is %ld\n",task.start_lba);
    // printf("lba cnt is %d\n",task.total_blocks);
    for (int i = 0; i < q_depth; i++) { do_io(&task); }
    while (1) {
        for (int  i = 0; i < nvme_cnt; i++)
        {
            spdk_nvme_qpair_process_completions(ns_entry[i]->qpair[thread_idx], 0);
        }
        if (task.finished_blocks == task.total_blocks) { break; }
    }
    unsigned long end = rdtscll();
    if (type == READ) {
        total_io_size += size;
        total_io_time += end - start;
    }
    // printf("io time: %f size %ld\n", (end - start)/2300.0, size);
}

thread_local std::vector<struct io_task*> io_tasks;

void io_callback_async(void* task, const struct spdk_nvme_cpl* cpl)
{
    if (spdk_nvme_cpl_is_error(cpl)) {
        printf("read error!\n");
        exit(1);
    }
    struct io_task* t = (struct io_task*)task;
    t->finished_blocks++;
    return;
}

void spdk_print_stat()
{
    INFO_PRINT("Total I/O size: %lu\n", total_io_size);
    INFO_PRINT("Total I/O time: %f\n", total_io_time/2.3/1000000000);
    INFO_PRINT("Total I/O CPU time: %f\n", cpu_cycle/2.3/1000000000);
    INFO_PRINT("I/O Bandwidth: %f GB/s\n", (double)total_io_size / total_io_time * 2.3);
    total_io_size = 0;
    total_io_time = 0;
    cpu_cycle = 0;
}