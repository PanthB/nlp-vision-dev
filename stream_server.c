#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <signal.h>
#include <pthread.h>

#define WIDTH           640
#define HEIGHT          480
#define BPP             2
#define FRAME_SIZE      (WIDTH * HEIGHT * BPP)
#define SERVER_PORT     5013
#define DURATION_SEC    600
#define VDMA_BASE       0x43000000
#define VDMA_RANGE      0x10000

#define UDP_HEADER_SIZE  12
#define UDP_PAYLOAD_MAX  1388
#define UDP_PACKET_MAX   (UDP_HEADER_SIZE + UDP_PAYLOAD_MAX)

#define S2MM_VDMACR     0x30
#define S2MM_VDMASR     0x34
#define S2MM_VSIZE      0xA0
#define S2MM_HSIZE      0xA4
#define S2MM_STRIDE     0xA8
#define S2MM_START_ADDR0 0xAC

static volatile uint32_t *vdma;
static volatile int running = 1;

static void sig_handler(int sig) { (void)sig; running = 0; }
static inline void vdma_wr(uint32_t off, uint32_t val) { vdma[off/4] = val; }
static inline uint32_t vdma_rd(uint32_t off) { return vdma[off/4]; }

static inline double time_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void sleep_ms(double ms) {
    long ns = (long)(ms * 1e6);
    struct timespec ts = { ns / 1000000000L, ns % 1000000000L };
    nanosleep(&ts, NULL);
}

#define RING_SIZE 8
static uint8_t *ring[RING_SIZE];
static volatile int ring_head = 0;
static volatile int ring_tail = 0;
static pthread_mutex_t ring_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  ring_cond = PTHREAD_COND_INITIALIZER;
static volatile int send_ok = 1;
static int g_udp_fd;
static struct sockaddr_in g_client_addr;
static socklen_t g_client_len = sizeof(g_client_addr);

/*
 * Fragment a frame into UDP packets and send.
 * Header: frame_number (4), packet_num (4), total_packets (4) — big-endian.
 * Matches receiver.py packet format.
 */
static int send_frame_udp(uint32_t frame_num, const uint8_t *frame, size_t frame_len) {
    size_t offset = 0;
    uint32_t total = (frame_len + UDP_PAYLOAD_MAX - 1) / UDP_PAYLOAD_MAX;
    uint8_t pkt[UDP_PACKET_MAX];

    for (uint32_t i = 0; i < total && send_ok; i++) {
        size_t chunk = frame_len - offset;
        if (chunk > UDP_PAYLOAD_MAX) chunk = UDP_PAYLOAD_MAX;

        pkt[0]  = (frame_num >> 24) & 0xFF;
        pkt[1]  = (frame_num >> 16) & 0xFF;
        pkt[2]  = (frame_num >>  8) & 0xFF;
        pkt[3]  = frame_num & 0xFF;
        pkt[4]  = (i >> 24) & 0xFF;
        pkt[5]  = (i >> 16) & 0xFF;
        pkt[6]  = (i >>  8) & 0xFF;
        pkt[7]  = i & 0xFF;
        pkt[8]  = (total >> 24) & 0xFF;
        pkt[9]  = (total >> 16) & 0xFF;
        pkt[10] = (total >>  8) & 0xFF;
        pkt[11] = total & 0xFF;
        memcpy(pkt + UDP_HEADER_SIZE, frame + offset, chunk);

        ssize_t n = sendto(g_udp_fd, pkt, UDP_HEADER_SIZE + chunk, 0,
                           (struct sockaddr *)&g_client_addr, g_client_len);
        if (n <= 0 || (size_t)n != UDP_HEADER_SIZE + chunk) return -1;
        offset += chunk;
    }
    return 0;
}

static void *send_thread_fn(void *arg) {
    (void)arg;
    while (running && send_ok) {
        pthread_mutex_lock(&ring_mtx);
        while (ring_head == ring_tail && running)
            pthread_cond_wait(&ring_cond, &ring_mtx);
        if (!running) { pthread_mutex_unlock(&ring_mtx); break; }
        int idx = ring_tail % RING_SIZE;
        uint32_t frame_num = (uint32_t)ring_tail;
        ring_tail++;
        pthread_mutex_unlock(&ring_mtx);

        if (send_frame_udp(frame_num, ring[idx], FRAME_SIZE) < 0) {
            fprintf(stderr, "[stream_server] sendto failed for frame %u\n", frame_num);
            send_ok = 0;
            break;
        }
        if (frame_num <= 1) {
            fprintf(stderr, "[stream_server] Frame %u sent OK.\n", frame_num);
            fflush(stderr);
        }
    }
    return NULL;
}

static void enqueue_frame(const volatile uint8_t *fb) {
    int space = RING_SIZE - (ring_head - ring_tail);
    if (space <= 0) return;
    int idx = ring_head % RING_SIZE;
    memcpy(ring[idx], (const void *)fb, FRAME_SIZE);
    pthread_mutex_lock(&ring_mtx);
    ring_head++;
    pthread_cond_signal(&ring_cond);
    pthread_mutex_unlock(&ring_mtx);
}

static void wait_for_frame_write_complete(const volatile uint8_t *fb) {
    volatile uint32_t *sentinel = (volatile uint32_t *)(fb + FRAME_SIZE - 4);
    uint32_t old_val = *sentinel;
    int timeout = 500000;
    while (timeout-- > 0) {
        uint32_t new_val = *sentinel;
        if (new_val != old_val) {
            usleep(200);
            return;
        }
    }
}

int main(int argc, char *argv[]) {
    fprintf(stderr, "[stream_server] vUDP-9b2c1f START\n");
    fflush(stderr);

    if (argc < 2) {
        fprintf(stderr, "Usage: sudo %s <buf0_phys>\n", argv[0]);
        return 1;
    }

    uint32_t buf0_phys = strtoul(argv[1], NULL, 16);
    fprintf(stderr, "[stream_server] Buffer: 0x%08x\n", buf0_phys);
    fflush(stderr);

    signal(SIGINT, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    int memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd < 0) { perror("open /dev/mem"); return 1; }

    vdma = (volatile uint32_t *)mmap(NULL, VDMA_RANGE,
        PROT_READ | PROT_WRITE, MAP_SHARED, memfd, VDMA_BASE);
    if (vdma == MAP_FAILED) { perror("mmap VDMA"); return 1; }

    volatile uint8_t *fb0 = (volatile uint8_t *)mmap(NULL, FRAME_SIZE,
        PROT_READ, MAP_SHARED, memfd, buf0_phys);
    if (fb0 == MAP_FAILED) { perror("mmap fb0"); return 1; }

    for (int i = 0; i < RING_SIZE; i++)
        ring[i] = (uint8_t *)malloc(FRAME_SIZE);

    fprintf(stderr, "[stream_server] Mapped OK.\n");
    fflush(stderr);

    uint32_t stride = WIDTH * BPP;
    vdma_wr(S2MM_VDMACR, 0x04);
    while (vdma_rd(S2MM_VDMACR) & 0x04);
    vdma_wr(S2MM_VDMASR, 0xFFFFFFFF);
    vdma_wr(S2MM_START_ADDR0, buf0_phys);
    vdma_wr(S2MM_STRIDE, stride);
    vdma_wr(S2MM_HSIZE, stride);
    vdma_wr(S2MM_VDMACR, 0x03);
    vdma_wr(S2MM_VSIZE, HEIGHT);
    usleep(200000);
    fprintf(stderr, "[stream_server] VDMA SR: 0x%05x\n", vdma_rd(S2MM_VDMASR));
    fflush(stderr);

    g_udp_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (g_udp_fd < 0) {
        perror("[stream_server] socket"); return 1;
    }
    int opt = 1;
    setsockopt(g_udp_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(SERVER_PORT);
    if (bind(g_udp_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("[stream_server] bind"); return 1;
    }
    fprintf(stderr, "[stream_server] UDP port %d bound. Waiting for client START packet...\n", SERVER_PORT);
    fflush(stderr);

    uint8_t buf[64];
    ssize_t n = recvfrom(g_udp_fd, buf, sizeof(buf), 0,
                         (struct sockaddr *)&g_client_addr, &g_client_len);
    if (n < 0) { perror("recvfrom"); return 1; }
    fprintf(stderr, "[stream_server] Client %s:%d connected (received %zd bytes). Sending frames.\n",
            inet_ntoa(g_client_addr.sin_addr), ntohs(g_client_addr.sin_port), n);
    fflush(stderr);

    pthread_t sender;
    pthread_create(&sender, NULL, send_thread_fn, NULL);

    double start_time = time_now();
    uint32_t frame_count = 0;

    while (running && send_ok) {
        if (time_now() - start_time >= DURATION_SEC) break;

        int timeout = 1000000;
        while (timeout-- > 0) {
            if (vdma_rd(S2MM_VDMASR) & 0x1000) break;
        }
        vdma_wr(S2MM_VDMASR, 0x1000);

        enqueue_frame(fb0);
        frame_count++;

        sleep_ms(25.0);
        wait_for_frame_write_complete(fb0);

        enqueue_frame(fb0);
        frame_count++;
    }

    running = 0;
    pthread_cond_signal(&ring_cond);
    pthread_join(sender, NULL);

    double total = time_now() - start_time;
    printf("Captured %u in %.1f s = %.1f FPS\n",
           frame_count, total, frame_count / total);

    for (int i = 0; i < RING_SIZE; i++) free(ring[i]);
    close(g_udp_fd);
    munmap((void *)vdma, VDMA_RANGE);
    munmap((void *)fb0, FRAME_SIZE);
    close(memfd);
    return 0;
}
