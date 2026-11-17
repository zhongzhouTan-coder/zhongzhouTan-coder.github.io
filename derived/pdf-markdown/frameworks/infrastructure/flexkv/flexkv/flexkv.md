# FlexKV: Flexible Index Offloading for Memory-Disaggregated Key-Value Store

Zhisheng Hu<sup>1</sup> Jiacheng Shen<sup>2</sup> Ming-Chang Yang<sup>1</sup>

<sup>1</sup>The Chinese University of Hong Kong <sup>2</sup>Duke Kunshan University

## Abstract

Disaggregated memory (DM) is a promising data center architecture that decouples CPU and memory into independent resource pools to improve resource utilization. Building on DM, memory-disaggregated key-value (KV) stores are adopted to efficiently manage remote data. Unfortunately, existing approaches suffer from poor performance due to two critical issues: 1) the overdependence on one-sided atomic operations in index processing, and 2) the constrained efficiency in compute-side caches. To address these issues, we propose FlexKV, a memory-disaggregated KV store with index proxying. Our key idea is to dynamically offload the index to compute nodes, leveraging their powerful CPUs to accelerate index processing and maintain high-performance compute-side caches. Three challenges have to be addressed to enable efficient index proxying on DM, i.e., the load imbalance across compute nodes, the limited memory of compute nodes, and the expensive cache coherence overhead. FlexKV proposes: 1) a rank-aware hotness detection algorithm to continuously balance index load across compute nodes, 2) a two-level CN memory optimization scheme to efficiently utilize compute node memory, and 3) an RPC-aggregated cache management mechanism to reduce cache coherence overhead. The experi mental results show that FlexKV improves throughput by up to 2.94× and reduces latency by up to 85.2%, compared with the state-of-the-art memory-disaggregated KV stores.

## 1 Introduction

Disaggregated memory (DM), a promising data center archi tecture proposed to improve resource efficiency, has gained significant attention from both industry [6, 35, 67, 79, 80] and academia [18, 39, 40, 60]. DM breaks the boundaries of monolithic servers by decoupling CPUs and memory into the compute pool (compute nodes, CNs) and the memory pool (memory nodes, MNs), connected via high-performance networks like RDMA [28, 54] and CXL [10, 36, 85, 86]. This architecture enables CPUs and memory to be flexibly allo cated, enhancing overall resource efficiency.

Key-value (KV) stores are emerging applications on DM [25, 33, 51, 62, 65]. Scalability, i.e., the ability to serve more clients without significant performance degradation, is a key requirement for all memory-disaggregated KV stores. However, existing approaches still struggle to achieve this goal due to the following two primary issues.

The overdependence on one-sided atomic operations in index processing. Existing memory-disaggregated KV stores extensively rely on one-sided atomic operations (i.e., CAS and FAA) to resolve concurrency conflicts during index processing [88]. Unfortunately, the extensive number of one-sided atomic operations generates a severe performance bottleneck on DMs implemented with both RDMA and CXL. Specifically, for RDMA-based DM, atomic operations with more complex semantics consume more processing power than read or write operations [31], making the processing units of RDMA network interface cards (RNICs) a bottleneck. Meanwhile, for CXL-based DM, atomic operations hinder system performance due to the expensive cross-node cache coherence protocol (i.e., CXL.cache in CXL 3.0) [10, 41, 48, 63].

The constrained efficiency in compute-side caches. Due to the significant performance gap between remote and local memory, using the local memory of CNs as caches is a common practice in many memory-disaggregated KV stores [33, 42, 46]. To reduce cross-node cache coherence overhead, existing designs either cache only the index (i.e., addresses of KV pairs) [45,62,65] or cache both the index and KV pairs but partition the ownership of cached data to individual CNs [33,42]. However, both approaches are suboptimal in terms of system performance. The former requires at least one remote memory access per request even on cache hits, while the latter introduces additional overhead from forwarding KV requests to designated nodes that own the requested data.

In this paper, we propose to offload part of the index to the compute pool to address the above issues simultaneously. Specifically, by offloading the index, CNs can help with index processing, converting expensive one-sided atomic operations into local CPU operations. Furthermore, during the offloaded index processing, CNs can efficiently cache KV pairs, enhancing compute-side cache efficiency. However, three challenges have to be addressed to achieve this idea in practice.

(1) The load imbalance across compute nodes. Index partitions may experience vastly different access frequencies due to workload skewness. When offloading them to CNs with static assignment, hot partitions can overload certain CNs while leaving others underutilized, degrading overall performance. Moreover, workload patterns are dynamic in production environments [7, 27, 37, 78], requiring continuous hotness tracking and dynamic reassignment. However, frequent reassignments cause significant overhead, making it challenging to balance load effectively.

(2) The limited memory of compute nodes. CNs have abundant CPUs but very limited memory [45, 59, 62], making it impractical to store the entire index locally in CNs, especially for large-scale production datasets [7, 11, 78]. Furthermore, offloading the index, whether fully or partially, to CNs introduces additional memory overhead, particularly from metadata needed for index operations. This further strains the already limited CN memory, interfering with user request processing and degrading system performance.

(3) The expensive coherence overhead of caching KV pairs. When KV pairs are cached across multiple CNs, main taining cache coherence becomes crucial for ensuring data consistency and correctness in the memory-disaggregated KV store. However, implementing a coherence protocol across network-connected CNs introduces significant overhead. First is the routine metadata maintenance overhead, i.e., the system must continuously manage and update cache metadata, such as the cache directory. Second is the bursty write invalidation overhead, i.e., updating a KV pair that is cached on multiple CNs triggers invalidations across those nodes. These operations substantially amplify the number of remote accesses, making caching KV pairs expensive.

To address the above challenges, we propose FlexKV, a memory-disaggregated KV store with index proxying that can dynamically offload the index to the compute pool and maintain efficient compute-side caches for KV pairs. First, to address load imbalance, FlexKV adopts a rank-aware hotness detection algorithm that continuously tracks partition access frequencies and triggers reassignments only when workload imbalance becomes significant. Second, to overcome lim ited CN memory, FlexKV employs a two-level CN memory optimization scheme: it uses slot-resolved index RPCs to eliminate heavy key-to-slot mappings on CNs, and introduces an adaptive index-cache splitting technique that efficiently partitions the limited CN memory between index and cache to maximize performance. Finally, to reduce cache coherence overhead, FlexKV leverages an RPC-aggregated cache man agement mechanism that piggybacks cache metadata maintenance (e.g., directory updates and hotness tracking) onto index processing, amortizing metadata maintenance costs while enabling selective caching of read-intensive KV pairs.

We implement FlexKV and evaluate its performance with YCSB [9] and Twitter [78] workloads. The results show that FlexKV improves throughput by up to 2.94× and reduces latency by up to 85.2%, compared with Clover [65], FUSEE [62], and Aceso [25], three state-of-the-art memorydisaggregated KV stores.

In summary, this paper makes the following contributions:

• We identify two key issues in existing approaches: 1) the overdependence on one-sided atomic operations in index processing, and 2) the constrained efficiency in compute-side caches. We propose that offloading the index to the compute pool can address both issues.

• We present FlexKV, a memory-disaggregated KV store with index proxying. By dynamically offloading the index to the compute pool, FlexKV prevents index pro cessing from becoming the performance bottleneck and achieves efficient compute-side caching for KV pairs.

• We implement FlexKV and conduct a thorough evalua-

![](flexkv/images/1b411c94808de8e7631d80b3e077ea27b6ccc32e63c3616a5c0d8c293e4ab48b.jpg)  
Figure 1: Index deployment in memory-disaggregated KV stores.  
tion, demonstrating the effectiveness of our design.

## 2 Background

## 2.1 Disaggregated Memory

In traditional data centers, the unit of deployment is monolithic servers, where CPUs and memory are allocated in fixed proportions. Such a server-centric approach results in significant resource underutilization [20, 43, 64]. On disaggregated memory (DM), monolithic servers are decoupled into compute nodes (CNs) and memory nodes (MNs). CNs have ample CPUs but limited memory, while MNs have abundant memory but only a few wimpy CPUs. Such a separation enables flexible, on-demand allocation of CPU and memory resources by adding the appropriate type of node to the cluster, thus achieving better resource utilization [18, 34, 53, 60, 72].

In a memory-disaggregated data center, CNs and MNs are connected through high-performance networks such as RDMA [54] and CXL [10]. Since CXL 3.0 devices are still under development, this paper focuses on RDMA-based DM, but the techniques discussed also consider future CXL-based DM (§4.6). RDMA provides applications with one-sided and two-sided verbs for accessing remote data. One-sided operations (e.g., WRITE, READ) enable direct memory access on remote nodes without remote CPU involvement, with onesided atomic operations like compare-and-swap (CAS) and fetch-and-add (FAA) for handling concurrent conflicts [88]. Two-sided operations (e.g., SEND, RECV) are similar to traditional sockets, requiring CPU participation at both ends.

## 2.2 Memory-Disaggregated KV Stores

## 2.2.1 Index Deployment on DM.

Based on the location of the index, existing memorydisaggregated KV stores can be classified into two categories.

(1) In monolithic metadata servers. For example, in KV stores like Clover [65] and SWARM-KV [51], while KV pairs are stored in MNs, the index is stored in additional monolithic metadata servers (MS), as shown in Figure 1(a). This design leads to significant resource waste on the monolithic servers managing the index [62]. Moreover, as all cluster-wide index operations are concentrated on a few monolithic servers, they can easily become the performance bottleneck.

(2) In memory nodes. Many memory-disaggregated KV stores choose to distribute the index across multiple MNs (e.g., Aceso [25], FUSEE [62], DINOMO [33]), as shown in Figure 1(b). Such an approach eliminates the need for monolithic servers and achieves full disaggregation, thus enhancing resource utilization. However, these approaches heavily rely on one-sided operations to read/write remote memory in MNs. The performance of this design is directly tied to the limited processing power of RNICs in the MNs, especially for high cost one-sided atomic operations. Since the number of RNICs for each MN is fixed and cannot be adjusted dynamically, they become a severe performance bottleneck under high concurrency, undermining disaggregated memory’s core promise: independent scaling of compute and memory.

![](flexkv/images/59e45e33b6d84799ceaf3eeb929c481788d791f557c4ecce179e91d66afd870b.jpg)  
Figure 2: Compute-side caching in memory-disaggregated KV stores. A client in CN 1 attempts to fetch the KV pair with key B.

Our choice: In both memory nodes and compute nodes. In this paper, we propose to distribute the index across both MNs and CNs (i.e., dynamically offloading a portion of the index from MNs to CNs), as shown in Figure 1(c). This design allows the index processing performance to scale with the expansion of either MNs or CNs, breaking the bottleneck.

## 2.2.2 Compute-Side Caching on DM.

Based on the design of compute-side caches, existing approaches can also be classified into two categories.

(1) Address-only cache. To minimize the overhead of maintaining cache coherence, many memory-disaggregated KV stores cache (e.g., Aceso [25], FUSEE [62], Clover [65]) only the index (i.e., addresses of KV pairs), as shown in Figure 2(a). When reading a KV pair through a cached address, the client verifies its correctness by checking a valid bit in the KV pair header. If the KV pair is invalid, the client invalidates the cached address. However, even on cache hits, this addressonly caching strategy requires at least one remote memory access to fetch the actual KV pair, limiting performance.

(2) Hybrid cache with ownership partitioning. Some memory-disaggregated KV stores adopt a hybrid caching strategy that caches both the index and KV pairs (e.g., DI-NOMO [33], DEX [42]), as shown in Figure 2(b). To minimize the overhead of ensuring cache coherence, they partition the ownership of data among CNs, where each CN handles the KV requests only for a given key range. Such an approach eliminates the need to invalidate caches on other CNs when one CN updates data. However, they adopt a client-CN-MN three-tier architecture. When a client initiates a KV request, if the CN responsible for the key is not local, the client must first forward the request to the designated CN, whose CPU then processes the subsequent operations and returns the result. As KV requests cannot be served by arbitrary CNs, this architecture exacerbates load imbalance and limits system performance due to the additional network overhead of forwarding, as demonstrated in our experiment (§5.3).

![](flexkv/images/b3e8560827f39fa0f199d2198341ba0d5a073d5ce25bb0f8835ee0178f1ac470.jpg)

![](flexkv/images/892902a15718906e3fc6d55da79b88c3e6b893145213a910958e44d417d6ff2d.jpg)  
Figure 3: The throughput and la- Figure 4: The effect of varying tency of different operations. the ratio of replaced operations.

Our choice: Hybrid cache with coherent sharing. In this paper, we propose a hybrid caching strategy where CNs share cached data with coherence maintained via a directory-based protocol, as shown in Figure 2(c). This design allows a KV pair to be cached by any CN, eliminating the MN-access overhead in address-only caching. Simultaneously, a KV request can be served by any CN, avoiding the load imbalance and for warding overhead inherent to ownership partitioning. Finally, we adopt a directory-based coherence protocol rather than a snoop-based protocol [16] because snoop-based protocols rely on broadcast messages, which limits system scalability and degrades performance as the number of CNs increases.

## 3 Motivation

In the following sections, we provide a deeper analysis of the differences between different schemes of index operations (i.e., one-sided CAS vs. proxied CAS), as well as between remote data accesses and local cache hits (i.e., one-sided read vs. local read), to further motivate our design choices.

## 3.1 One-Sided CAS vs. Proxied CAS

Existing memory-disaggregated KV stores typically rely on one-sided atomic operations to resolve concurrency conflicts. As mentioned in §1, these one-sided atomic operations are more complex and incur greater overhead on the processing units within RNICs. In these KV stores, frequent invocation of one-sided atomic operations can easily make the MN RNICs a performance bottleneck. In contrast, letting the CN CPUs handle concurrency conflicts using two-sided RPCs (RDMA\_SEND&RECV) together with local atomic instructions (LOCAL\_CAS) can significantly improve performance, as they are more lightweight and avoid accessing the MNs.

Figure 3 shows the throughput of RDMA\_CAS, RDMA\_WRITE, RDMA\_SEND&RECV, and LOCAL\_CAS, all at 8B granularity, with 200 clients (i.e., the same setup as in §5.1), as well as their P50 and P99 latencies in the single-client-per-CN case. Among these operations, RDMA\_CAS and RDMA\_WRITE are issued from

CNs to MNs, while RDMA\_SEND&RECV occurs between CNs. RDMA\_WRITE, RDMA\_SEND&RECV, and LOCAL\_CAS achieve 10.1×, 19.5×, and 177.1× higher throughput than RDMA\_CAS, respectively, while also exhibiting lower latency, particu larly LOCAL\_CAS. Notably, this performance gap holds across RNIC generations (e.g., ConnectX-4 [74]/5 [57]/6 [31]), since the relatively high overhead of one-sided atomics remains un changed. These results highlight the superior efficiency of these operations compared to the inefficient RDMA\_CAS, suggesting that replacing RDMA\_CAS with them under high load can provide substantial performance improvement. Figure 4 shows the throughput clearly increases when gradually replacing RDMA\_CAS with the combination of RDMA\_SEND&RECV and LOCAL\_CAS, validating this observation.

Based on the above observation and to address the issue described in §2.2.1, FlexKV proposes the index proxying that dynamically offloads the index to CNs. Each CN can act as the proxy for an exclusive set of index partitions, managing these partitions in its local memory via LOCAL\_CAS. All index operations targeting these partitions are routed to this proxy via two-sided RPCs (implemented with RDMA\_SEND&RECV). This scheme avoids a large portion of CN-to-MN RDMA\_CAS operations, alleviating network congestion at MN RNICs.

## 3.2 One-Sided Read vs. Local Read

Figure 3 and 4 also show the performance of RDMA\_READ and LOCAL\_READ (each triggering a single memcpy()), all at 128B granularity. Notably, LOCAL\_READ achieves 38.2× higher throughput than RDMA\_READ, suggesting that caching KV pairs locally could allow read requests to complete with much higher efficiency. However, as discussed in §2.2.2, ex isting memory-disaggregated KV stores either do not cache KV pairs or rely on ownership partitioning, both of which limit performance. To address this issue, proxies in FlexKV, in addition to handling RPCs, also maintain directory-based caches for KV pairs. This allows any CN to cache any KV pair, facilitating clients to directly serve read requests from the local cache without triggering any remote operations.

## 4 The FlexKV Design

## 4.1 Overview

We present FlexKV, a memory-disaggregated KV store with index proxying. The main idea of index proxying is to dy namically offload the index to CNs, with each CN acting as a proxy responsible for managing an exclusive portion of the global index, leveraging powerful CN CPUs to improve index processing performance and enable efficient compute-side caching. Figure 5 illustrates the overview of FlexKV. In each CN, clients represent upper-layer applications accessing the KV store, the proxy handles RPCs for the local index, which is loaded from MNs and consists of several index partitions, and the local cache caches the global index and KV pairs. The memory pool consists of multiple MNs, holding the index and KV pairs. The manager coordinates cluster-level operations such as hotness detection and parameter tuning.

![](flexkv/images/1e85cfc6fde5d2c5b104c836b10035b1d0350df0125ce40e17bf7e6098ded600.jpg)  
Figure 5: The overview of FlexKV.

Request Processing. FlexKV provides a standard KV store API [13,25,51,55,62,65], including INSERT, UPDATE, SEARCH, and DELETE. We refer to SEARCH as read requests and INSERT, UPDATE, DELETE as write requests. The processing of a KV request is broadly divided into two parts: KV pair access and index processing. For KV pair access, clients directly read from or write to MNs via one-sided operations. For index processing, FlexKV employs a hybrid approach (Figure 5): if the corresponding index partition resides on MNs, the index processing is performed directly by the client using one-sided operations, following existing work [25, 62]. Otherwise, if the partition is offloaded to a CN (i.e., index proxying), the index processing is forwarded to the CN’s proxy via twosided RPCs. The proxy first writes the index updates to MNs via RDMA\_WRITE to ensure recoverability, then commits the updates to its local index via LOCAL\_CAS. The full protocol and correctness analysis of FlexKV is presented in §4.5.

RPC Types. In FlexKV, proxies handle two-sided RPCs for different purposes. RPCs sent from clients to proxies for executing index operations are referred to as index RPCs, which are the primary RPCs in our design. Other types of RPCs are used for auxiliary tasks (e.g., parameter tuning).

Challenges. As analyzed in §1, applying index proxying confronts several challenges. To address load imbalance, FlexKV adopts a rank-aware hotness detection algorithm to evenly assign index partitions to CNs (§4.2). To overcome limited CN memory, FlexKV employs a two-level CN memory optimization scheme: it uses slot-resolved index RPCs to reduce the space overhead of the local index (§4.3.1), and uses an adaptive index-cache splitting technique to partition CN memory between the local index and local cache (§4.3.2). To reduce cache coherence overhead, FlexKV leverages an RPC-aggregated cache management mechanism (§4.4).

## 4.2 Rank-Aware Hotness Detection

To enable index proxying, the first step is to assign index partitions to CNs. However, this poses a challenge: many workloads exhibit significant skewness [7, 9, 27, 37, 78], causing index partitions to experience vastly different access frequencies. With static or random assignment, hot partitions can overload certain CNs while leaving others underutilized, degrading overall performance. Moreover, workload patterns are dynamic in production environments, and frequent reassignments incur non-trivial overhead, making this problem particularly challenging (Challenge 1).

To address this challenge, FlexKV adopts a rank-aware hotness detection algorithm, as shown in Algorithm 1. The design is guided by two key strategies. First, to ensure load balancing across CNs, it employs a rank-based partition assignment scheme. Second, to prevent oscillations caused by minor hotness fluctuations, it calculates a rank-level displacement score.

Rank-Based Partition Assignment. FlexKV separates the global index into $P = 2 ^ { x }$ partitions using the first x bits of the key hash $( i . e . , x = 1 3$ in our implementation) and tracks the hotness of each partition. To distribute these partitions evenly, the manager sorts all partitions in descending order of hotness (Line 8). Let C be the number of CNs. We group these sorted partitions into $R = P / C$ contiguous ranks, each containing C partitions. Rank 1 contains the C hottest partitions, rank 2 contains the next C hottest, and so on. Each CN is then assigned exactly one partition from each rank. Consequently, this assignment forms a partition-to-CN map (Figure 6, left), where each CN maintains a hot-to-cold partition list ordered by rank. When a proxy selects partitions to offload based on the memory budget, it scans this list from head to tail, so the hottest partitions are offloaded first. This assignment not only balances the global load but also facilitates the adap tive index-cache splitting technique in §4.3.2, where proxies dynamically adjust the number of partitions to offload.

Rank-Level Displacement Score. FlexKV is designed to react only to significant workload shifts. We define the displacement score D to quantify the degree of change in the global hotness distribution. Let $R _ { n e w } ( p ) \in \{ 1 , ~ . . . , R \}$ denote the rank of partition p under the currently observed hotness ordering (Line 13), and let $R _ { o l d } ( p )$ denote its rank under the previous ordering. The rank-level displacement score is then calculated as the sum of the rank changes for all partitions: $\begin{array} { r } { D = \sum _ { p = 1 } ^ { P } | R _ { n e w } ( p ) - R _ { o l d } ( p ) | } \end{array}$ . This metric is appropriate because same-rank partitions share the same offloading priority level. Thus, minor hotness fluctuations that do not change a partition’s rank can be safely ignored.

Combining the two strategies above, the manager executes Algorithm 1 every ∆ seconds (i.e., ∆ = 1 in our implementa tion). To track hotness, clients maintain a 4-byte per-partition access counter over a ∆-second sliding window. Every ∆ sec onds, the manager RDMA\_READs these counters from CNs and aggregates them to obtain the global hotness (Line 7). To determine when to trigger a reassignment, we compare D against a baseline B, which represents the expected displacement of a random reshuffle. Let X and Y be independently uniform on $\{ 1 , . . . , R \}$ . The per-partition expected rank displacement under a random reshuffle is $E [ | X - Y | ] = ( R ^ { 2 } - 1 ) / ( 3 R ) .$ , a classical formula in mathematics. Hence, the baseline for all partitions is $B = P \cdot E \big [ | X - Y | \big ] = C ( R ^ { 2 } - 1 ) / 3$ . We trigger reassignment when the current ranking shifts by at least 25% of a fully random reshuffle (Line 16), i.e., $D \geq 0 . 2 5 B .$ . The 25% threshold is determined empirically to offer a good trade-off between stability and responsiveness.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Rank-aware hotness detection.

1: // P: number of partitions
2: // C: number of CNs
3: // access_count[p][c]: partition p's access count at CN c
4: //  $R_{old}[p]$ : previous rank of partition p
5: procedure HOTNESSDETECT(P, C, access_count,  $R_{old}$ )
6:    for  $p \in \{1..P\}$  do ▷ Collect access counts
7:    hotness[p] ←  $\sum_{c=1}^{C} access\_count[p][c]$ 
8:    sorted_partition ← SORTPARTITION(hotness)
9:    $R \leftarrow P/C$ 
10:    for  $r \in \{1..R\}$  do ▷ Assign new ranks
11:    for  $i \in \{(r-1)C + 1..rC\}$  do
12:    $p \leftarrow sorted\_partition[i]$ 
13:    $R_{new}[p] \leftarrow r$ 
14:    $D \leftarrow \sum_{p=1}^{P} |R_{new}[p] - R_{old}[p]|$ ▷ Compute displacement
15:    $B \leftarrow C(R^{2}-1)/3$ ▷ Compute baseline
16:    if  $D \geq 0.25B$  then ▷ Trigger reassignment if needed
17:    TRIGGERREASSIGNMENT()
18:    $R_{old} \leftarrow R_{new}$
</div>

![](flexkv/images/757b88d6b4bc4b98f1dac5f55e497df9965d6e9b3f76a9ea3e9f929562d06509.jpg)  
Figure 6: The rank-based index partition assignment.

Atomic Partition Reassignment. To prevent any partition from being served by multiple nodes simultaneously during reassignment, FlexKV uses a two-phase pause-resume protocol. Each CN keeps two partition-to-CN maps: an active map and a staging map that holds the new assignment. (1) Pause. The manager installs the new assignment into all staging maps via RDMA\_WRITE, then notifies all CNs through two-sided RPCs to initiate pausing. Upon receiving the RPC, each CN compares its active and staging maps to identify the partitions whose ownership changes, pauses serving new requests for these partitions, and clears the local cache to ensure coherence. After quiescing these partitions, it returns an ACK. (2) Resume. Once all ACKs arrive, the manager sends resume notifications. Each CN switches its staging map to active, loads newly assigned partitions from MNs, and resumes service for the affected partitions. In our experiment (§5.3), each reassignment round finishes within 3-5 ms, well below typical multi-second workload shift timescales [7, 27, 37, 78].

![](flexkv/images/5e3ef95c7e6e478a15e85edd8c7156eb9c3c67d682b06ca1dce7f04a84011183.jpg)  
Figure 7: The slot-resolved index RPC mechanism.

## 4.3 Two-Level CN Memory Optimization

In the DM architecture, CNs have abundant CPUs but very limited memory. A CN typically has only about 10% of the memory of a single MN [42, 65, 81], or even less [33, 45, 71]. Index proxying offloads part of the index from MNs to CNs increases pressure on the already limited CN memory and can even degrade performance (Challenge 2).

To address this challenge, FlexKV employs the two-level CN memory optimization scheme. At the first level, slot-resolved index RPCs are used to reduce the inherent space overhead of the local index. At the second level, an adaptive index cache splitting technique is applied to efficiently partition CN memory between the local index and local cache.

## 4.3.1 Slot-Resolved Index RPC

In many DM index designs, the index alone cannot pinpoint a key’s exact index slot without reading the corresponding KV pair. For example, SMART [46] and RACE [89] store only partial keys or fingerprints, so a key identifies only candidate index slots. The system must fetch the KV pairs referenced by those slots from MNs to confirm the correct one before performing updates. Consequently, in FlexKV, a proxy cannot resolve a key’s index slot from the key when serving index RPCs. An alternative is to maintain a direct mapping from keys to slot addresses within the CN’s local index, but this map requires storing keys. With variable-sized keys (e.g., strings), the memory overhead becomes substantial, making the local index prohibitively memory-intensive.

To make the local index lightweight, FlexKV issues slotresolved RPCs. This design is enabled by the fact that, for fault tolerance, the proxy synchronizes every local index update to the MNs’ index (§4.5). This allows clients to resolve a key’s index slot address at the MNs before sending index RPCs to the proxy, thereby avoiding the proxy’s own slot lookup and the memory-intensive key-to-slot map. Furthermore, to reduce index lookup traffic to MNs, FlexKV employs different index-resolution strategies for write and read requests.

Slot-Resolved Write Requests. For write requests (i.e., INSERT/UPDATE/DELETE), FlexKV has clients resolve the target index slot before issuing index RPCs to the proxy, as shown in Figure 7. Specifically, the client RDMA\_READs the MNs’ index by key to find candidate index slots (❶), RDMA\_READs the referenced KV pairs to confirm the correct one (❷), and then sends that slot address along with the RPC (❸). The proxy can then update the designated slot directly without a key-based lookup. Notably, this approach does not add client-side burden and maintains correctness because it follows the original one-sided index protocol exactly, except for replacing the client’s RDMA\_CAS with a LOCAL\_CAS at the proxy, while other procedures remain identical. To accelerate slot resolution, FlexKV embeds the slot address in each key’s cache entry (i.e., the KV pair or its address), so the full resolution path is taken only when the local cache misses.

Fast-Path Read Requests. For read requests (i.e., SEARCH), clients bypass slot resolution and issue index RPCs directly, with the proxy returning all candidate index slots based on the key (①). The client then fetches the KV pairs from these slots and finds the one whose key matches (②). Since slot collisions are rare, the proxy typically returns only one candidate slot. With this design, most read requests are efficiently served by the proxy, reducing index lookup traffic to MNs.

## 4.3.2 Adaptive Index-Cache Splitting

Once the local index is introduced on CNs, CN memory must be shared between the local index and the local cache. Under tight CN memory budgets, allocating memory between these two components is challenging. Overprovisioning one component can starve the other, sometimes leading to performance even worse than a simpler design that only has the local cache. Moreover, the optimal memory split is workload-dependent and time-varying (e.g., skew shifts, read-write ratio changes), making static partitioning inherently suboptimal.

To address this issue, FlexKV employs an adaptive indexcache splitting technique to search for a near-optimal memory split under dynamic workloads. Specifically, it 1) introduces a unified index-offload ratio across all CNs to reduce the search space, and 2) applies a throughput-guided knob that dynamically tunes this ratio in response to workload shifts.

Unified Index-Offload Ratio. FlexKV uses a sampling-based approach that adjusts the memory split based on observed throughput, offering robustness and implementation simplicity. However, a naive sampler can be slow to converge due to the large search space: with C CNs, there are C split variables to tune. To collapse the search space to a single variable, FlexKV introduces a unified index-offload ratio, defined as the ratio between offloaded partitions and total partitions in the per-CN hot-to-cold partition list (Figure 6). All CNs use the same ratio to determine the memory allocated to their local index, with the remaining memory assigned to the local cache. Since index partitions at the same rank across CNs exhibit similar access frequencies (§4.2), applying this cluster-wide index-offload ratio naturally balances the load.

Throughput-Guided Knob. FlexKV tunes the index-offload ratio via a periodic, hill-climbing knob guided by sampled throughput (Algorithm 2), leveraging the generally unimodal relationship between index-offload ratio and throughput, as we demonstrate in §5.4. Let i ∈ [0,1] denote the fraction of the index offloaded to CNs. Every ∆ seconds, the manager perturbs i by a fixed step size δ, updating $i = i + s \cdot \delta$ according to the current search direction $s \in \{ - 1 , 1 \}$ . This knob is stateful: it initializes with i = 0 and s = 1 (Line 2), and flips direction at the first probe of each round if throughput immediately degrades (Line 10). It continues stepping while throughput improves. After reaching a candidate maximum, the round terminates once the next two probes both underperform the current best. New rounds are triggered by workload shifts (Line 5), i.e., either a change of at least 10% in the read-write ratios (e.g., from 1:9 to 2:8) or a partition reassignment (§4.2). In our implementation, we set ∆ = 1 and δ = 0.1 (i.e., the ratio is adjusted by 10% per second) to prevent overreacting to transient fluctuations. While FlexKV currently optimizes for throughput only, the knob can be extended to incorporate other SLOs, such as latency, which is left for future work.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 Throughput-guided knob.

1: procedure KNOB()
2:    $i \leftarrow 0, s \leftarrow 1$   ▷ Initialize knob state
3:    while true do
4:    STARTROUND()
5:    wait until WORKLOADSHIFT()
6: procedure STARTROUND()
7:    $i_{best} \leftarrow i, T_{best} \leftarrow \text{SAMPLE}(i, \Delta), U_{best} \leftarrow 0$ 
8:    $T_{first} \leftarrow \text{SAMPLE}(i + s \cdot \delta, \Delta)$ 
9:    if $T_{first} &lt; T_{best}$ then
10:    $s \leftarrow -s$   ▷ Flip search direction
11:    while $U_{best} &lt; 2$ do
12:    $i \leftarrow i + s \cdot \delta, T \leftarrow \text{SAMPLE}(i, \Delta)$ 
13:    if $T \leq T_{best}$ then
14:    $U_{best} \leftarrow U_{best} + 1$ 
15:    else   ▷ Find new candidate
16:    $i_{best} \leftarrow i, T_{best} \leftarrow T, U_{best} \leftarrow 0$ 
17:    $i \leftarrow i_{best}$

Local Cache Local Index
Addresses KV Pairs Index Buffer Meta Buffer
Cache Directory
Metadata Entry Sharer List Write Counter Read Counter
32 bit 16 bit 16 bit
</div>

![](flexkv/images/d00fddb4697a3b570501809abe1de62d4ed03478baed8cce37d41eb6b75421fb.jpg)  
Figure 8: The memory layout of the compute node.

## 4.4 RPC-Aggregated Cache Management

To accelerate data access, FlexKV implements compute-side caching for both the index and KV pairs, with the latter managed by a directory-based coherence protocol (§2.2.2). However, this protocol suffers from prohibitive coherence overheads, stemming from both routine metadata maintenance and bursty write invalidations (Challenge 3).

To address this challenge, FlexKV employs an RPCaggregated cache management mechanism. The core insight is to leverage the flexibility of two-sided RPCs to manage cache directories and prioritize caching read-intensive KV pairs. Unlike one-sided operations that only support simple semantics (e.g., read/write), RPCs handled by CN CPUs can execute arbitrary logic. FlexKV exploits this by piggybacking multiple fine-grained, coherence-related operations (i.e., cache directory updates, KV hotness tracking) onto index RPCs. Since these operations are executed in the same network round-trip as index processing, they incur no additional network overhead and thus reduce the routine metadata maintenance cost. Furthermore, the piggybacked KV hotness tracking enables selective caching of read-intensive KV pairs, avoiding caching write-intensive ones that would be quickly invalidated, thus mitigating the impact of bursty write invalidations.

![](flexkv/images/aaafbf006d70127794db59b324eabfbe0c1d8f56a6a2194a5ff34c0487f5d203.jpg)  
Figure 9: The RPC-aggregated cache management mechanism. When handling an index RPC for a write request, the proxy increments the write counter, invalidates cached copies via a directory (i.e., the sharer list), and then commits the index slot.

Figure 8 illustrates the CN memory layout. The local cache is used by clients to accelerate KV request execution, while the local index is managed by the proxy. For the local cache, it stores either the address or the KV pair of a key, but never both. Cached addresses and KV pairs are managed under a unified FIFO eviction policy, which incurs minimal CPU overhead while providing sufficient performance. More advanced eviction strategies are beyond the scope of this paper. For the local index, it is further divided into the index buffer and the metadata buffer. The index buffer stores index partitions loaded from MNs, while the metadata buffer holds a metadata entry for each key within these partitions. In each metadata entry, the 32-bit sharer list bitmap serves as the cache directory that tracks which CNs have cached this KV pair. For larger-scale clusters, the fixed per-key bitmaps can be replaced with more compact formats (e.g., sparse directory format [76]), which is orthogonal to our work. The 16-bit write counter and read counter increment by 1 when the proxy handles an index RPC from a write request and from a read request, respectively. Since 16 bits can easily max out at 65535, the proxy shifts both the write and read counters right by 2 bits when this overflow happens. Although this reduces precision, it preserves the recent ratio between the write and read counters, which serves as the key indicator for identifying read-intensive KV pairs that merit caching.

Piggybacked Metadata Maintenance. FlexKV piggybacks numerous coherence-related operations onto index RPCs to reduce the routine metadata maintenance cost, as shown in Figure 9. Specifically, when the proxy handles an index RPC, it simultaneously updates the read/write counter of the key to track its read/write hotness, and updates the cache directory to record the sharer<sup>1</sup> if necessary. All of these tasks can be easily finished within the index RPC processing step. Furthermore, since each partition is offloaded to a single CN, all index RPCs for a key are handled by the same CN, meaning the read-write statistics and the cache directory are naturally centralized, eliminating the need for additional aggregation.

![](flexkv/images/7444cd79fe117fed4c7730aa23c149ff464ca5e198059849b813d635337f60a4.jpg)  
Figure 10: The workflows of different KV requests in FlexKV.

Read-Write Hotness-Aware Caching. FlexKV selectively caches read-intensive KV pairs to mitigate the impact of bursty write invalidations. When handling an index RPC, in addition to updating the read/write counter, the proxy cal culates the ratio between the write and read counters of the key. If this ratio falls below a threshold (i.e., 0.25 in our implementation), the KV pair is considered cache-worthy. The proxy then marks the RPC sender as a sharer in the sharer list, enabling it to cache the KV pair, and notifies the sender via the RPC response to store the KV pair in its local cache. During the above process, for UPDATE or DELETE requests, the proxy invalidates previous cached copies by issuing two-sided RPCs to CNs in the old sharer list (if non-empty). Proxies of these sharer CNs delete their cached copies and check for in-flight RPCs on the same key to prevent them from caching stale KV pairs upon receiving the RPC responses, ensuring consistency in the presence of race conditions.

When a client accelerates a read request through the local cache, it bypasses the proxy (Figure 10), leading to the loss of some read hotness. To address this, clients locally accumulate the read-counter increments when processing read requests without sending index RPCs. The next time an RPC for the same key is issued, the accumulated increments are piggybacked to the proxy for updating. If no RPC is sent for an extended period, or if the accumulated increments exceed a threshold (i.e., 32), the client triggers a dedicated RPC to flush the read-counter increments to the proxy.

## 4.5 Put It All Together

This section presents the complete framework of FlexKV, integrating the three previously presented design components. Index Structure. Without loss of generality, following existing studies [25, 62, 65, 83, 84], we use a hash table [89] as a case study for indexing KV pairs. The hashtable is divided into subtables, which serve as the granularity for index partitioning among proxies in our implementation. Each subtable contains contiguous buckets, each consisting of contiguous 8-byte slots. A slot comprises a 48-bit address field for the

KV pair’s location in the memory pool, an 8-bit length field for the KV pair size, and an 8-bit fingerprint field for preliminary key matching during lookups. The index uses 8-byte CAS operations to atomically modify index slots.

Memory Management. To ensure high read performance [25, 62, 65, 89], the out-of-place update scheme is employed for KV pairs, where each write request allocates a new KV pair before modifying the index. To manage KV pair space, FlexKV employs a two-level method [25, 62]. Clients first request coarse-grained memory blocks (e.g., 16 MB) from MNs when their free space is insufficient, then allocate fine-grained KV pairs from these blocks upon receiving write requests.

Workflows. Figure 10 presents the FlexKV workflows through which clients execute the four types of KV requests: INSERT, UPDATE, SEARCH, and DELETE. As FlexKV’s design mainly targets proxy-managed index partitions, we concentrate on these specific workflows. The workflows for other index partitions follow identical protocols to existing one-sided RDMA based KV stores [25, 62], and are omitted for brevity.

(1) Write. For a write request (INSERT/UPDATE/DELETE), the client first writes the new KV pair to MNs while resolving the index slot address. The resolution can be skipped when the local cache is hit (§4.3.1). After this, the client issues an index RPC to the proxy, which then invalidates previous caches: (i) for cached addresses, the proxy clears the previously committed KV pair’s valid bit to invalidate (§2.2.2); (ii) for cached KV pairs, the proxy sends RPCs to those CNs on the sharer list to invalidate (§4.4). In parallel, the proxy writes the new index slot to the MNs’ index for fault tolerance. Finally, it commits the new slot to its local index via LOCAL\_CAS (commit point) and replies to the client.

(2) Read. For a read request (SEARCH), the client first checks the local cache. If a cached KV pair exists, the client immediately returns the result. If a cached address exists, the client performs an RDMA\_READ to fetch the KV pair from MNs and verifies the match. If both miss, the client sends an index RPC to the proxy to obtain the address of the latest committed KV pair (§4.3.1), which is then retrieved via RDMA\_READ.

When handling index RPCs for write requests, the proxy uses a local key-to-lock map to ensure only one write request per key is processed at a time. Concurrent writes to keys already in process will fail immediately, as in CAS. Since in-flight RPCs are few, the map size remains negligible.

Fault Tolerance. We assume that the network is reliable and no Byzantine failures [24] occur, following previous work [25, 51, 62, 83]. (1) For the manager, it only performs periodic parameter tuning tasks without maintaining any persistent state. Its failure is safe and does not affect clients. (2) For MNs, currently, FlexKV employs primary-backup replication [32] to ensure MN data safety, where each write is replicated to multiple MNs (3-way in our evaluation). (3) For CNs, FlexKV treats all CN data as disposable. Loss of local caches is safe, as they contain temporary data. Loss of local indexes is also safe since proxies write all updates to MNs before modifying the local indexes. As MN failures can be effectively handled by existing solutions [30, 47, 51, 62], we focus on CN failures.

Upon detecting a CN failure (via a membership service [19, 26]), surviving CNs clear their local caches, since the failed CN’s cache directory is lost and cached KV pairs it managed can no longer be validated for coherence. These CNs also update their partition-to-CN maps (§4.2) to redirect all requests targeting the failed CN’s index partitions to MNs, allowing clients to continue accessing those partitions via one-sided operations. The entire response completes within 3-5ms in our experiments, ensuring minimal service disrup tion. When the failed CN restarts, the manager re-offloads the corresponding index partitions to it using the same procedure as the atomic partition reassignment (§4.2).

Linearizability and Correctness. FlexKV guarantees linearizability [23] by treating the proxy’s LOCAL\_CAS as the atomic commit point. We demonstrate correctness by analyzing the visibility of KV pair versions across three possible read paths: ❶ cached KV pair, ❷ cached address, and ❸ index RPC.

(1) Path convergence. Before LOCAL\_CAS, the proxy strictly enforces cache invalidations, effectively disabling paths ❶ and ❷. This forces all concurrent reads to converge to path ❸, making the proxy the single source of truth.

(2) Atomic visibility switch. With read traffic converged, LOCAL\_CAS acts as an atomic boundary. Before execution, the proxy’s local index references the old KV pair; immediately after, the new one. Note that the new index slot written to MNs before LOCAL\_CAS is used solely for slot resolution (§4.3.1) and does not expose the uncommitted write.

Finally, if a proxy fails after writing the new index slot to MNs but before executing LOCAL\_CAS, clients fall back to accessing MNs directly, making the uncommitted write visible. However, this can be treated as a redo of an incomplete write [62], preserving linearizability.

Garbage Collection. (1) For KV pairs. Since FlexKV employs out-of-place KV pair updates, it needs to reclaim stale KV pairs. When clients execute UPDATE or DELETE requests, they add the old KV pairs to a free list in each CN, allowing their space to be reused by subsequent allocations, similar to previ ous work [65]. (2) For index slots. DELETE frees a key’s index slot, but other CNs may still hold cached slot addresses for this key (§4.3.1). FlexKV uses a lease-based method to ensure safety: each cached slot address is valid for at most $T _ { l e a s e }$ time and is renewed on each successful operation on that key. We repurpose the 48-bit address field in index slots: the first bit serves as a valid bit, while the remaining 47 bits store either the actual address (when valid=1) or a DELETE timestamp $T _ { d e l e t e }$ (when valid=0). During DELETE, instead of clearing the slot, we set the valid bit to 0 and record $T _ { d e l e t e }$ . Subsequent INSERT can safely reuse an invalid slot once the current time exceeds $T _ { d e l e t e } + T _ { l e a s e } \times ( 1 + \delta )$ , where δ accounts for the bounded cluster-wide clock drift (up to $\ S = 1 0 ^ { - 4 } \left[ 1 5 , 3 8 , 5 2 \right] )$ In our implementation, we set $T _ { l e a s e } = 2 0 0 \mathrm { m s }$

Other Optimizations. FlexKV incorporates several optimiza tions: (1) UD-based RPC. We use unreliable datagram (UD) QPs for two-sided RPCs to avoid QP explosion between clients and proxies, with timeouts ensuring reliability [29]. (2) RDMA optimizations. We adopt standard techniques including coroutines, inline writes, doorbell batching, and selective signaling [28, 62, 71]. To ensure fairness, we also add these optimizations to all baselines in our evaluation (§5).

## 4.6 Discussion

Generality of FlexKV to Other KV Stores. The design of FlexKV is general-purpose. First, offloading index partitions to CNs is applicable to other index data structures, e.g., SMART [46], Sherman [71], and CHIME [45], since onesided RDMA operations can be directly replaced with local memory accesses. Once the index is offloaded, the proposed compute-side caching and memory partitioning designs become readily applicable to these systems.

Compatibility of FlexKV over CXL. We consider CXL 3.0, which enables hardware-based memory sharing similar to RDMA. Although CXL 3.0 provides one-sided atomics via a MESI-like coherence protocol [3, 10], recent studies reveal performance challenges [48, 63, 77, 85]. Thus, FlexKV’s techniques remain valuable to future CXL-based DM. For inter-CN RPCs, FlexKV can leverage CXL-based RPC designs, as explored in some pioneering work [48, 49].

## 5 Evaluation

## 5.1 Experimental Setup

Testbed. We conduct experiments with 23 physical machines (i.e., 20 CNs and 3 MNs) on the Apt cluster of CloudLab [12], with the manager colocated on the first CN. Each machine features a 56Gbps Mellanox ConnectX-3 IB RNIC connected to a 56Gbps Mellanox SX6036G switch, two 8-core Intel E5-2650v2 CPUs, and 64GB of DRAM. We present the sensitivity results for different CN-MN ratios in §5.4.

Comparisons. We compare FlexKV with three state-of-the-art memory-disaggregated KV stores: Clover [65], FUSEE [62], and Aceso [25]. Clover places the index on additional monolithic servers (i.e., metadata servers), requiring separate accesses for index operations (Figure 1(a)). To ensure fairness, we allocate an additional physical machine to Clover as the metadata server. FUSEE and Aceso, conversely, place the index in MNs, with clients performing index operations via one-sided RDMA (Figure 1(b)). The key difference is that FUSEE employs replication for index fault tolerance, requiring multiple RDMA\_CAS to multiple replicas for each index update, while Aceso adopts a checkpointing mechanism that requires only a single RDMA\_CAS. To demonstrate the impact of ownership partitioning [33, 42], we implement FlexKV-OP, a variant of FlexKV adopting the ownership partitioning scheme. In FlexKV-OP, each request must first be forwarded to a designated CN, determined by hashing, for processing. Workloads. We employ YCSB benchmarks [9] and Twitter workloads [78] for our evaluation. The YCSB bench marks use four core workloads: A (write-intensive, 50% UPDATE and 50% SEARCH), B (read-intensive, 5% UPDATE and 95% SEARCH), C (read-only, 100% SEARCH), and D (insertinvolved, 5% INSERT and 95% SEARCH). By default, all YCSB workloads employ skewed key generation following a Zipfian distribution $( \mathsf { \alpha } = 0 . 9 9 ^ { 2 } )$ [4], as per the YCSB standard con figuration. We also evaluate the performance under uniform YCSB workloads in §5.2. The Twitter workloads comprise 54 real-world traces collected from production clusters, with varying read-write ratios, KV pair sizes, and skewness.

![](flexkv/images/44e0172af0288a95c98d5bc91d34d5f22843bd306d3a12cb8368a9c73d8cca66.jpg)  
(a) YCSB A

![](flexkv/images/e0fad0101d7c56ae6f1609cf1ba4a47b330dc8622e0e5967753e068714865e83.jpg)  
(b) YCSB B

![](flexkv/images/f86eae201acc75d0a4e3c86f22bf450dbf1a575eb71ddec32371f3abf9a941fb.jpg)  
(c) YCSB C

![](flexkv/images/4f901441544babef98965d2ecc8a9afdcb3c75590891eeb87e57da407292a91b.jpg)  
(d) YCSB D

Figure 11: The throughput under YCSB workloads.  
![](flexkv/images/1d20bbba56a3be9041d599a44e676a94ca701f3fc908b010224c6ed093ee4747.jpg)  
(a) YCSB A

![](flexkv/images/5fae20dfa2cf20fa1187cfb9f522e052f90a3ba111f286400b2fd24bf4a91192.jpg)  
(b) YCSB B

![](flexkv/images/13ae82fd0141b170962a26f89c11ac031387157adebfb8e93d8d87388dea44e4.jpg)  
(c) YCSB C

![](flexkv/images/d0c2492a9937719334bae84ecb848edd69b24be35d9e7233cfa559337198be92.jpg)  
(d) YCSB D  
Figure 12: The latency CDF under YCSB workloads.

Parameters. Unless otherwise specified, we use 200 clients evenly distributed across 20 CNs, with each client running 8 coroutines. Each MN provides 56 GB of memory, for a total of 168 GB. All systems use 3-way replication or an equivalent fault-tolerance setup. Each CN has 64 MB of memory, roughly 5% of the working-set size, matching DM’s configuration. For all YCSB experiments, we bulk-load 10 million KV pairs before issuing any requests, and the KV pair size is set to 128 B, which is representative of real-world workloads [7, 9, 11, 78]. Sensitivity results for varying KV pair sizes and CN memory limits are in §5.4. To ensure fairness, points with the same index across curves in a figure use identical configurations (the number of CNs, clients, and coroutines).

## 5.2 Performance Comparison

YCSB Skewed Workloads. We present the performance of FlexKV, Aceso, FUSEE, and Clover under YCSB workloads.

![](flexkv/images/078f2f832139779efa490618d2f337904d4785cdd128353d9edb102f3c84faf3.jpg)

![](flexkv/images/cea6ced4383f1120c53c631de830fa3b0d70c343b065f1ac68825caa8336de61.jpg)  
Figure 13: The throughput under Figure 14: The throughput under different UPDATE-SEARCH ratios. uniform YCSB workloads.

To generate throughput-client curves, we vary the number of clients from 20 to 200. The results are shown in Figure 11. FlexKV achieves the highest throughput across all workloads, with peak throughput improvements over the second-best system of 2.31× (YCSB A), 1.34× (YCSB B), 1.37× (YCSB C), and 1.31× (YCSB D). Aceso and FUSEE are limited by their reliance on expensive RDMA\_CAS for index operations, making MN RNICs a bottleneck. Clover suffers from frequent accesses that overload its metadata server. In contrast, FlexKV leverages index proxying to balance index operations across all nodes and employs KV pair caching to achieve superior performance on different workloads.

We further evaluate request latency for all systems with 200 clients, presenting the cumulative distribution functions (CDFs) in Figure 12. Clover achieves the lowest P50 latency because, while it bottlenecks at the metadata server, clients that hit the address cache bypass the metadata server and access MNs directly. Since Clover’s throughput is lower, its MN RNICs experience less congestion and can service RDMA operations quickly. For P99 latency, FlexKV achieves the lowest values across YCSB A/B/C/D, reducing P99 latency by 85.2%, 36.4%, 4.1%, and 36.9% compared to the second-best system, respectively. On YCSB C, FlexKV’s P99 latency is close to Aceso’s, as FlexKV’s 37% higher throughput increases network pressure and offsets its latency advantage.

![](flexkv/images/e262b6d8dedb9bda33b9f1103013777fc127fabbd09e5a3ab817c15abc0f024f.jpg)  
Figure 15: The throughput under Twitter workloads.

![](flexkv/images/4599a3d0c168f4ac613c0752989f2d54574823b62ca99ecf4e166cbae2815ee0.jpg)  
Figure 16: The ablation study on FlexKV.

![](flexkv/images/47e0d337dfecd8d1da5c1b365c3a35b1cc339dc5e41b67b150fc3a7b9b2f1384.jpg)  
Figure 17: The effect of owner ship partitioning on FlexKV.

Different Update-Search Ratios. Figure 13 shows the throughput under different UPDATE-SEARCH ratios. Throughput decreases for all systems as the proportion of UPDATE increases, since UPDATE is more complex than SEARCH. FlexKV consistently achieves the highest throughput across all ratios. In read-intensive scenarios, FlexKV mainly benefits from effective KV pair caching, while in write-intensive scenarios, from index proxying that offloads index processing to CNs.

YCSB Uniform Workloads. We modify the key distribution in the YCSB workloads from Zipfian to uniform, and Figure 14 shows the results. FlexKV’s throughput drops be cause the uniform access pattern lowers local cache hit ratios. Moreover, under the uniform distribution, the offloaded index partitions become less frequently accessed than in the skewed case, which reduces the effectiveness of index proxying.

Twitter Workloads. Figure 15 shows the throughput under Twitter workloads, with clusters ordered by FlexKV’s throughput in descending order. Across all clusters, FlexKV achieves 1.03–2.94× higher throughput than the second-best system. For some workloads, the gains are modest, as they are either uniform (e.g., cluster 35 with Zipfian α = 0) or involve large KV pairs, causing all systems to be limited by network bandwidth (e.g., cluster 50). In contrast, workloads with high skewness and read-intensive patterns benefit substantially. For example, cluster 1 (α = 2.68, 99% reads) sees significant improvement from FlexKV’s KV pair caching. These results highlight FlexKV’s effectiveness in real-world environments.

## 5.3 System-level Analysis

Ablation Study. Figure 16 shows the impact of FlexKV’s key techniques on performance under YCSB workloads. Base is the basic version with only address caching. +Proxy enables the index proxies, statically offloading the first 20% index partitions to CNs, improving throughput by 14.5% on average across four workloads. +Rank Hotness identifies and offloads the hottest index partitions to CNs, improving throughput by 11.9% on average. +KV Cache enables KV pair caching, improving throughput by 6.1% on average. +Adaptive Split dynamically adjusts the index-offload ratio, improving throughput by 15.2% on average. Together, these techniques incrementally build up to the full performance of FlexKV.

![](flexkv/images/8fa5c88c22d69a2618053314d1c65d72ed1515297a6d65c32afffd95b4b09142.jpg)  
Figure 18: The throughput under dynamic workload pattern.

![](flexkv/images/bcc996b9a28f1c70ee69bf9c2e68237a170edb6ba3116723a7ac17bc32ee4152.jpg)  
Figure 19: The load distribution across CNs.

![](flexkv/images/c8cfd2bac39bbae3c1fd92b21f2f2b376358688e099900007f04ad83198fe4e5.jpg)  
Figure 20: The throughput under different proxy thread numbers.

Cost of Ownership Partitioning. As mentioned in §5.1, we implement FlexKV-OP, a variant of FlexKV that employs ownership partitioning similar to DINOMO [33] and DEX [42], where each KV request is forwarded to a designated CN based on its key for subsequent processing. This design lets each CN manage a distinct key range, maintaining local caches without coherence overhead. However, as shown in Figure 17, ownership partitioning reduces FlexKV’s throughput by 24.9% on average across four workloads, due to the extra network overhead caused by request forwarding.

Dynamic Workload Pattern. To demonstrate FlexKV’s adaptability to dynamic workload patterns, we run YCSB B (5% UPDATE) for 15 seconds, then switch to YCSB A (50% UPDATE), as shown in Figure 18. The performance is sampled every 10 ms. At 1 s, the manager runs Algorithm 1 for index partition reassignment. At 2 s, the manager runs Algorithm 2 to search for a near-optimal index-offload ratio. At 7 s, it finds the ratio of 30%. At 8 s, throughput drops slightly as cache evictions increase CPU overhead when local caches fill up. At 15 s, the UPDATE ratio changes to 50%, causing throughput to drop to 12.5 Mops/s. At 16 s, the manager detects the workload pattern change and re-runs Algorithm 2 to search for a new index-offload ratio. At 22 s, it finds the new ratio of 60%, improving throughput to 17.3 Mops/s. This demonstrates FlexKV’s ability to adapt to workload changes and maintain an appropriate index-cache memory split. Throughout the process, each index partition reassignment or index-offload ratio adjustment takes 3.24-5.26 ms, causing minimal disruption. Effect of Load Balancing. To demonstrate the effect of the rank-aware hotness-detection algorithm on load balancing, Figure 19 shows the load distribution across CNs under YCSB A and reports the coefficient of variation (CV) as the balancing metric. Enabling the Algorithm 1 increases the total index operations offloaded to CNs by 14.8% and reduces the CV by

Table 1: The performance breakdown under YCSB workloads.

<table><tr><td rowspan="2">Workload</td><td rowspan="2">Offload Ratio (%)</td><td colspan="2">Hit Ratio (%)</td><td colspan="3">SEARCH Latency (μs)</td></tr><tr><td>KV</td><td>Addr</td><td>KV</td><td>Addr</td><td>Other</td></tr><tr><td>YCSB A</td><td>60</td><td>0.1</td><td>10.4</td><td>2.3</td><td>24.1</td><td>54.1</td></tr><tr><td>YCSB B</td><td>30</td><td>10.1</td><td>24.1</td><td>1.9</td><td>23.6</td><td>52.3</td></tr><tr><td>YCSB C</td><td>80</td><td>18.9</td><td>30.6</td><td>2.2</td><td>16.5</td><td>42.8</td></tr><tr><td>YCSB D</td><td>50</td><td>15.5</td><td>31.3</td><td>2.3</td><td>23.3</td><td>47.4</td></tr></table>

## 42.7%, significantly improving load balancing.

Performance Breakdown. We present the index-offload ratios, cache hit ratios, and average SEARCH latencies with 200 clients in Table 1. For write-intensive YCSB A, the indexoffload ratio is high (60%) to reduce the pressure of frequent RDMA\_CAS on MNs, while the value cache hit ratio remains low (0.1%) as FlexKV avoids caching frequently modified KV pairs. For read-intensive YCSB C, the index-offload ratio reaches 80% because more index partitions managed by proxies enable more KV pairs to be cached, achieving a higher value cache hit ratio (18.9%). YCSB B exhibits a lower indexoffload ratio (30%) as it is neither write-intensive nor has good caching locality. The latency breakdown shows that KV-pair cache hits incur only 2µs due to local data structure overhead (e.g., cache lookup and eviction), while address cache hits require one RDMA\_READ with about 20µs latency. When both caches miss, latency increases to about 50µs.

## 5.4 Sensitivity Analysis

Impact of Proxy Thread Number. In FlexKV, each proxy runs multiple threads to process index RPCs concurrently. Figure 20 shows throughput under YCSB workloads with different numbers of proxy threads. Throughput first increases with the number of threads, peaks at 2 threads, and then de clines due to resource contention (e.g., lock contention on multithreaded data structures and RNIC cache thrashing from more QPs [29, 31, 58]). We set the default number of proxy threads to 2. Considering CNs’ abundant CPUs and FlexKV’s up to 2.94× performance improvement, this modest 20% extra CPU usage is acceptable. Notably, a single proxy thread already achieves 71.8–90.2% of the peak throughput, indicating the CPU overhead of index proxying is small.

Impact of KV Pair Size. Figure 21 shows the throughput under KV pair sizes ranging from 128B to 1024B, increasing in 128B steps. As the KV size grows, throughput of all systems gradually decreases and their performance gap narrows due to bandwidth saturation. Nevertheless, FlexKV consistently achieves the highest throughput among the four systems.

Impact of CN-MN Ratio. While previous experiments use only 3 MNs, here we vary the CN-MN ratio from 3:20 to 10:13 to evaluate FlexKV’s scalability, as shown in Figure 22. Throughput of FlexKV, Aceso, and FUSEE first increases and then decreases. The initial increase is due to more MN RNICs, which mitigate the MN-side network bottleneck. The subse quent decrease occurs because each client reaches its CPU limit, and fewer CNs reduce the total number of clients. For Clover, performance remains limited by the metadata server, so varying CN-MN ratios has little effect.

![](flexkv/images/c153f40071db4c6b11f15dafe0930dc3bdaf4f5281679d26c0d464a98c7af5fe.jpg)  
Figure 21: The throughput under different KV pair sizes.

![](flexkv/images/45275a4607ae31cfed830d9ea195f4362dccf5521e601444d259081f369455aa.jpg)  
Figure 22: The throughput under different CN-MN ratios.

![](flexkv/images/032de932cd3ce60a216808bea900590a0bf4dca923c722c002bba5b52ed5a3a9.jpg)  
Figure 23: The throughput under different CN memory limits.

![](flexkv/images/a4ead436d99035bd262eea4937858fa90ec8953d057bb5c3371508c9d9010879.jpg)  
Figure 24: The throughput under different index-offload ratios.

Impact of CN Memory Limit. Figure 23 shows the throughput under CN memory limits ranging from 0MB to 128MB, increasing in 16MB steps. Throughput generally increases with larger CN memory, especially for read-intensive workloads. For the write-intensive YCSB A, only FlexKV shows significant improvement, as the other systems are limited by RDMA\_CAS or the metadata server. FUSEE shows little improvement because it prefetches index hash buckets even when cached addresses hit to reduce RTT, which causes readamplification and leaves FUSEE bandwidth-bound. In contrast, Aceso only fetches buckets on cache misses [25].

Impact of Index-Offload Ratio. We disable FlexKV’s knob (Algorithm 2) to fix the index-offload ratio, and manually vary it to evaluate performance. As shown in Figure 24, under YCSB workloads, throughput first increases and then slightly decreases as the offload ratio grows. This indicates a roughly unimodal relationship between the index-offload ratio and throughput, motivating Algorithm 2 to find a near-optimal ratio. It also shows that placing the index entirely in MNs or fully offloading it to CNs is suboptimal, while FlexKV’s partial offloading strategy achieves the best performance by fully utilizing all hardware resources in the cluster (Figure 1).

## 6 Related Work

Disaggregated Memory. Existing studies on DM include software-based approaches, such as modifying operating systems [2, 8, 56, 60, 66, 70, 75] and runtimes [5, 21, 68, 69, 73] for seamless integration, as well as efforts to optimize data structures [14, 45, 59, 71, 85, 87] and specific applications like in-memory KV stores [1, 25, 33, 42, 51, 62] and transactional storage systems [44, 74, 83, 84]. On the other hand, hardwarebased approaches [36, 39, 50, 63] and hardware-software co-

designed methods [22, 34, 72] are being explored to enhance functionality. FlexKV, an advanced KV store designed for DM, can also benefit from these low-level optimizations.

Memory-Disaggregated KV Stores. Recently, many KV stores targeting DM architectures have emerged, such as Clover [65], FUSEE [62], DINOMO [33], Aceso [25], and SWARM-KV [51], each with its unique design focus: Clover reduces monetary and energy costs, FUSEE disaggregates metadata management, DINOMO leverages ownership par titioning for cache efficiency, Aceso uses erasure coding to save space, and SWARM-KV trades space for one-RTT oper ations. Finally, FlexKV focuses on scalable index processing and flexible KV pair caching to improve performance.

Memory-Disaggregated Index Processing. In memory disaggregated KV stores, index processing are handled in various ways. Clover [65] relies on monolithic metadata servers, which limits resource efficiency and can become the performance bottleneck. Aceso [25] and FUSEE [62] let CN CPUs manage the index in MNs remotely via onesided RDMA. DEX [42] also uses one-sided RDMA but can offload some operations to MN CPUs for local execution, requiring MN compute power. DINOMO [33] relies entirely on MN CPUs, demanding significant MN compute power. In contrast, FlexKV allows CN CPUs to either manage the index remotely via one-sided RDMA or act as the proxies for local execution, without imposing any compute overhead on MNs. Memory-Disaggregated Caching. Ditto [61] is a memory disaggregated caching system for backup storage. It stores cached data in MNs and performs distributed eviction via one-sided RDMA. In contrast, FlexKV stores cached data in CNs and performs local eviction via CN CPUs. DiFache [82], a concurrent work with FlexKV, is a general-purpose caching system for DM applications. It stores cached data in CNs but keeps cache directory in MNs, which is managed via onesided RDMA. In contrast, FlexKV stores both cached data and cache directory in CNs, and directory management is piggybacked on index RPCs, reducing extra network overhead.

## 7 Conclusion

This paper presents FlexKV, a memory-disaggregated KV store with index proxying. FlexKV offloads the index to CNs dynamically and introduces KV pair caching across CNs, addressing the bottleneck at MN RNICs. Evaluation shows it improves throughput by up to 2.94× and reduces latency by up to 85.2%, compared with the state-of-the-art systems.

## References

[1] Ahmed Alquraan, Sreeharsha Udayashankar, Viren dra Marathe, Bernard Wong, and Samer Al-Kiswany. LoLKV: The logless, linearizable, RDMA-based Key-Value storage system. In 21st USENIX Symposium on

Networked Systems Design and Implementation (NSDI 24), pages 41–54, Santa Clara, CA, April 2024. USENIX Association.

[2] Emmanuel Amaro, Christopher Branner-Augmon, Zhihong Luo, Amy Ousterhout, Marcos K. Aguilera, Aurojit Panda, Sylvia Ratnasamy, and Scott Shenker. Can far memory improve job throughput? In Proceedings of the Fifteenth European Conference on Computer Systems, EuroSys ’20, Heraklion, Greece, 2020. Association for Computing Machinery.

[3] Gal Assa, Michal Friedman, and Ori Lahav. A programming model for disaggregated memory over cxl. ArXiv, abs/2407.16300, 2024.

[4] L. Breslau, Pei Cao, Li Fan, G. Phillips, and S. Shenker. Web caching and zipf-like distributions: evidence and implications. In IEEE INFOCOM ’99. Conference on Computer Communications. Proceedings. Eighteenth Annual Joint Conference of the IEEE Computer and Communications Societies. The Future is Now (Cat. No.99CH36320), volume 1, pages 126–134 vol.1, 1999.

[5] Irina Calciu, M. Talha Imran, Ivan Puddu, Sanidhya Kashyap, Hasan Al Maruf, Onur Mutlu, and Aasheesh Kolli. Rethinking software runtimes for disaggregated memory. In Proceedings ofthe 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, ASPLOS ’21, page 79–92, Virtual, USA, 2021. Association for Computing Machinery.

[6] Wei Cao, Yingqiang Zhang, Xinjun Yang, Feifei Li, Sheng Wang, Qingda Hu, Xuntao Cheng, Zongzhi Chen, Zhenjun Liu, Jing Fang, Bo Wang, Yuhui Wang, Haiqing Sun, Ze Yang, Zhushi Cheng, Sen Chen, Jian Wu, Wei Hu, Jianwei Zhao, Yusong Gao, Songlu Cai, Yunyang Zhang, and Jiawang Tong. Polardb serverless: A cloud native database for disaggregated data centers. In Proceedings ofthe 2021 International Conference on Management ofData, SIGMOD ’21, page 2477–2489, Virtual Event, China, 2021. Association for Computing Machinery.

[7] Zhichao Cao, Siying Dong, Sagar Vemuri, and David H.C. Du. Characterizing, modeling, and benchmarking RocksDB Key-Value workloads at facebook. In 18th USENIX Conference on File and Storage Technologies (FAST 20), pages 209–223, Santa Clara, CA, February 2020. USENIX Association.

[8] Lei Chen, Shi Liu, Chenxi Wang, Haoran Ma, Yifan Qiao, Zhe Wang, Chenggang Wu, Youyou Lu, Xiaobing Feng, Huimin Cui, Shan Lu, and Harry Xu. A tale of two paths: Toward a hybrid data plane for efficient Far-Memory applications. In 18th USENIX Symposium on

Operating Systems Design and Implementation (OSDI 24), pages 77–95, Santa Clara, CA, July 2024. USENIX Association.

[9] Brian F. Cooper, Adam Silberstein, Erwin Tam, Raghu Ramakrishnan, and Russell Sears. Benchmarking cloud serving systems with ycsb. In Proceedings of the 1st ACM Symposium on Cloud Computing, SoCC ’10, page 143–154, Indianapolis, Indiana, USA, 2010. Association for Computing Machinery.

[10] CXL. Compute express link 3.0, 2025. https://computeexpresslink.org/wp-content/ uploads/2024/02/CXL-3.0-Specification.pdf.

[11] Siying Dong, Andrew Kryczka, Yanqin Jin, and Michael Stumm. Evolution of development priorities in keyvalue stores serving large-scale applications: The RocksDB experience. In 19th USENIX Conference on File and Storage Technologies (FAST 21), pages 33–49. USENIX Association, February 2021.

[12] Dmitry Duplyakin, Robert Ricci, Aleksander Maricq, Gary Wong, Jonathon Duerig, Eric Eide, Leigh Stoller, Mike Hibler, David Johnson, Kirk Webb, Aditya Akella, Kuangching Wang, Glenn Ricart, Larry Landwe ber, Chip Elliott, Michael Zink, Emmanuel Cecchet, Snigdhaswin Kar, and Prabodh Mishra. The design and operation of CloudLab. In 2019 USENIX Annual Technical Conference (USENIX ATC 19), pages 1–14, Renton, WA, July 2019. USENIX Association.

[13] Brad Fitzpatrick. Distributed caching with memcached. Linux journal, 2004(124):5, 2004.

[14] Jian Gao, Youyou Lu, Minhui Xie, Qing Wang, and Jiwu Shu. Citron: Distributed range lock management with one-sided RDMA. In 21st USENIX Conference on File and Storage Technologies (FAST 23), pages 297–314, Santa Clara, CA, February 2023. USENIX Association.

[15] Jian Gao, Qing Wang, and Jiwu Shu. ShiftLock: Mitigate one-sided RDMA lock contention via handover. In 23rd USENIX Conference on File and Storage Technologies (FAST 25), pages 355–372, Santa Clara, CA, February 2025. USENIX Association.

[16] Vasilis Gavrielatos, Antonios Katsarakis, Arpit Joshi, Nicolai Oswald, Boris Grot, and Vijay Nagarajan. Scaleout ccnuma: exploiting skew with strongly consistent caching. In Proceedings of the Thirteenth EuroSys Con ference, EuroSys ’18, Porto, Portugal, 2018. Association for Computing Machinery.

[17] Jim Gray, Prakash Sundaresan, Susanne Englert, Ken Baclawski, and Peter J. Weinberger. Quickly generating billion-record synthetic databases. SIGMOD Rec., 23(2):243–252, May 1994.

[18] Juncheng Gu, Youngmoon Lee, Yiwen Zhang, Mosharaf Chowdhury, and Kang G. Shin. Efficient memory disaggregation with infiniswap. In 14th USENIX Symposium on Networked Systems Design and Implementation (NSDI 17), pages 649–667, Boston, MA, March 2017. USENIX Association.

[19] Rachid Guerraoui, Antoine Murat, Javier Picorel, Athanasios Xygkis, Huabing Yan, and Pengfei Zuo. uKharon: A membership service for microsecond applications. In 2022 USENIX Annual Technical Conference (USENIX ATC 22), pages 101–120, Carlsbad, CA, July 2022. USENIX Association.

[20] Jing Guo, Zihao Chang, Sa Wang, Haiyang Ding, Yihui Feng, Liang Mao, and Yungang Bao. Who limits the resource efficiency of my datacenter: an analysis of alibaba datacenter traces. In Proceedings of the International Symposium on Quality ofService, IWQoS ’19, Phoenix, Arizona, 2019. Association for Computing Machinery.

[21] Zhiyuan Guo, Zijian He, and Yiying Zhang. Mira: A program-behavior-guided far memory system. In Proceedings ofthe 29th Symposium on Operating Systems Principles, SOSP ’23, page 692–708, Koblenz, Germany, 2023. Association for Computing Machinery.

[22] Zhiyuan Guo, Yizhou Shan, Xuhao Luo, Yutong Huang, and Yiying Zhang. Clio: a hardware-software codesigned disaggregated memory system. In Proceedings of the 27th ACM International Conference on Architectural Supportfor Programming Languages and Operating Systems, ASPLOS ’22, page 417–433, Lausanne, Switzerland, 2022. Association for Computing Machinery.

[23] Maurice P. Herlihy and Jeannette M. Wing. Linearizability: a correctness condition for concurrent objects. ACM Trans. Program. Lang. Syst., 12(3):463–492, July 1990.

[24] Chi Ho, Robbert Van Renesse, Mark Bickford, and Danny Dolev. Nysiad: Practical protocol transformation to tolerate byzantine failures. In 5th USENIX Symposium on Networked Systems Design and Implementation (NSDI 08), San Francisco, CA, April 2008. USENIX Association.

[25] Zhisheng Hu, Pengfei Zuo, Yizou Chen, Chao Wang, Junliang Hu, and Ming-Chang Yang. Aceso: Achieving efficient fault tolerance in memory-disaggregated keyvalue stores. In Proceedings ofthe ACM SIGOPS 30th Symposium on Operating Systems Principles, SOSP ’24, page 127–143, Austin, TX, USA, 2024. Association for Computing Machinery.

[26] Patrick Hunt, Mahadev Konar, Flavio Paiva Junqueira, and Benjamin C. Reed. Zookeeper: Wait-free coordination for internet-scale systems. In Proceedings of the 2010 USENIX Annual Technical Conference, USENIXATC 2010, Boston, MA, USA, June 23-25, 2010. USENIX Association, 2010.

[27] Xin Jin, Xiaozhou Li, Haoyu Zhang, Robert Soulé, Jeongkeun Lee, Nate Foster, Changhoon Kim, and Ion Stoica. Netcache: Balancing key-value stores with fast in-network caching. In Proceedings of the 26th Sym posium on Operating Systems Principles, SOSP ’17, page 121–136, Shanghai, China, 2017. Association for Computing Machinery.

[28] Anuj Kalia, Michael Kaminsky, and David G. Andersen. Design guidelines for high performance RDMA systems. In 2016 USENIX Annual Technical Conference (USENIX ATC 16), pages 437–450, Denver, CO, June 2016. USENIX Association.

[29] Anuj Kalia, Michael Kaminsky, and David G. Andersen. FaSST: Fast, scalable and simple distributed transactions with Two-Sided (RDMA) datagram RPCs. In 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), pages 185–201, Savannah, GA, November 2016. USENIX Association.

[30] Antonios Katsarakis, Vasilis Gavrielatos, M.R. Siavash Katebzadeh, Arpit Joshi, Aleksandar Dragojevic, Boris Grot, and Vijay Nagarajan. Hermes: A fast, fault tolerant and linearizable replication protocol. In Pro ceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems, ASPLOS ’20, page 201–217, Lausanne, Switzerland, 2020. Association for Computing Machinery.

[31] Xinhao Kong, Jingrong Chen, Wei Bai, Yechen Xu, Mahmoud Elhaddad, Shachar Raindel, Jitendra Padhye, Alvin R. Lebeck, and Danyang Zhuo. Understanding RDMA microarchitecture resources for performance isolation. In 20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23), pages 31–48, Boston, MA, April 2023. USENIX Association.

[32] Leslie Lamport, Dahlia Malkhi, and Lidong Zhou. Ver tical paxos and primary-backup replication. In Proceedings of the 28th ACM Symposium on Principles of Distributed Computing, PODC ’09, page 312–313, Calgary, AB, Canada, 2009. Association for Computing Machinery.

[33] Sekwon Lee, Soujanya Ponnapalli, Sharad Singhal, Marcos K. Aguilera, Kimberly Keeton, and Vijay

Chidambaram. Dinomo: An elastic, scalable, highperformance key-value store for disaggregated persistent memory. Proc. VLDB Endow., 15(13):4023–4037, September 2022.

[34] Seung-seob Lee, Yanpeng Yu, Yupeng Tang, Anurag Khandelwal, Lin Zhong, and Abhishek Bhattacharjee. Mind: In-network memory management for disaggregated data centers. In Proceedings of the ACM SIGOPS 28th Symposium on Operating Systems Principles, SOSP ’21, page 488–504, Virtual Event, Germany, 2021. Association for Computing Machinery.

[35] Guoliang Li, Wengang Tian, Jinyu Zhang, Ronen Grosman, Zongchao Liu, and Sihao Li. Gaussdb: A cloud-native multi-primary database with computememory-storage disaggregation. Proc. VLDB Endow., 17(12):3786–3798, August 2024.

[36] Huaicheng Li, Daniel S. Berger, Lisa Hsu, Daniel Ernst, Pantea Zardoshti, Stanko Novakovic, Monish Shah, Samir Rajadnya, Scott Lee, Ishwar Agarwal, Mark D. Hill, Marcus Fontoura, and Ricardo Bianchini. Pond: Cxl-based memory pooling systems for cloud platforms. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS 2023, page 574–587, Vancouver, BC, Canada, 2023. Association for Computing Machinery.

[37] Jialin Li, Jacob Nelson, Ellis Michael, Xin Jin, and Dan R. K. Ports. Pegasus: Tolerating skewed workloads in distributed storage with In-Network coherence directories. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20), pages 387–406. USENIX Association, November 2020.

[38] Yuliang Li, Gautam Kumar, Hema Hariharan, Hassan Wassel, Peter Hochschild, Dave Platt, Simon Sabato, Minlan Yu, Nandita Dukkipati, Prashant Chandra, and Amin Vahdat. Sundial: Fault-tolerant clock synchronization for datacenters. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20), pages 1171–1186. USENIX Association, November 2020.

[39] Kevin Lim, Jichuan Chang, Trevor Mudge, Parthasarathy Ranganathan, Steven K. Reinhardt, and Thomas F. Wenisch. Disaggregated memory for expansion and sharing in blade servers. In Proceedings of the 36th Annual International Symposium on Computer Architecture, ISCA ’09, page 267–278, Austin, TX, USA, 2009. Association for Computing Machinery.

[40] Kevin Lim, Yoshio Turner, Jose Renato Santos, Alvin AuYoung, Jichuan Chang, Parthasarathy Ranganathan,

and Thomas F. Wenisch. System-level implications of disaggregated memory. In IEEE International Sympo sium on High-Performance Comp Architecture, pages 1–12, 2012.

[41] Jinshu Liu, Hamid Hadian, Yuyue Wang, Daniel S. Berger, Marie Nguyen, Xun Jian, Sam H. Noh, and Huaicheng Li. Systematic cxl memory characterization and performance analysis at scale. In Proceedings ofthe 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS ’25, page 1203–1217, Rot terdam, Netherlands, 2025. Association for Computing Machinery.

[42] Baotong Lu, Kaisong Huang, Chieh-Jan Mike Liang, Tianzheng Wang, and Eric Lo. Dex: Scalable range in dexing on disaggregated memory. Proc. VLDB Endow., 17(10):2603–2616, August 2024.

[43] Chengzhi Lu, Kejiang Ye, Guoyao Xu, Cheng-Zhong Xu, and Tongxin Bai. Imbalance in the cloud: An analy sis on alibaba cluster trace. In 2017 IEEE International Conference on Big Data (Big Data), pages 2884–2892, 2017.

[44] Haodi Lu, Haikun Liu, Yujian Zhang, Zhuohui Duan, Xiaofei Liao, Hai Jin, and Yu Zhang. Fast distributed transactions for rdma-based disaggregated memory. In 2025 USENIX Annual Technical Conference (USENIX ATC 25), pages 943–958, Santa Clara, CA, July 2025. USENIX Association.

[45] Xuchuan Luo, Jiacheng Shen, Pengfei Zuo, Xin Wang, Michael R. Lyu, and Yangfan Zhou. Chime: A cache efficient and high-performance hybrid index on disaggregated memory. In Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles, SOSP ’24, page 110–126, Austin, TX, USA, 2024. Association for Computing Machinery.

[46] Xuchuan Luo, Pengfei Zuo, Jiacheng Shen, Jiazhen Gu, Xin Wang, Michael R. Lyu, and Yangfan Zhou. SMART: A High-Performance adaptive radix tree for disaggregated memory. In 17th USENIX Symposium on Oper ating Systems Design and Implementation (OSDI 23), pages 553–571, Boston, MA, July 2023. USENIX Association.

[47] Xuhao Luo, Ramnatthan Alagappan, and Aishwarya Ganesan. Splitft: Fault tolerance for disaggregated datacenters via remote memory logging. In Proceedings of the Nineteenth European Conference on Computer Systems, EuroSys ’24, page 590–607, Athens, Greece, 2024. Association for Computing Machinery.

[48] Teng Ma, Zheng Liu, Chengkun Wei, Jialiang Huang, Youwei Zhuo, Haoyu Li, Ning Zhang, Yijin Guan, Dimin Niu, Mingxing Zhang, and Tao Ma. HydraRPC: RPC in the CXL era. In 2024 USENIX Annual Technical Conference (USENIX ATC 24), pages 387–395, Santa Clara, CA, July 2024. USENIX Association.

[49] Suyash Mahar, Ehsan Hajyasini, Seungjin Lee, Zifeng Zhang, Mingyao Shen, and Steven Swanson. Telepathic datacenters: Efficient and high-performance rpcs using shared cxl memory. In 15th Annual Non-Volatile Memories Workshop, 2024.

[50] Hasan Al Maruf, Hao Wang, Abhishek Dhanotia, Johannes Weiner, Niket Agarwal, Pallab Bhattacharya, Chris Petersen, Mosharaf Chowdhury, Shobhit Kanaujia, and Prakash Chauhan. Tpp: Transparent page placement for cxl-enabled tiered-memory. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, ASPLOS 2023, page 742–755, Vancouver, BC, Canada, 2023. Association for Computing Machinery.

[51] Antoine Murat, Clément Burgelin, Athanasios Xygkis, Igor Zablotchi, Marcos Kawazoe Aguilera, and Rachid Guerraoui. Swarm: Replicating shared disaggregatedmemory data in no time. In Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles, SOSP ’24, page 24–45, Austin, TX, USA, 2024. Association for Computing Machinery.

[52] Ali Najafi and Michael Wei. Graham: Synchronizing clocks by leveraging local clock properties. In 19th USENIX Symposium on Networked Systems Design and Implementation (NSDI 22), pages 453–466, Renton, WA, April 2022. USENIX Association.

[53] Vlad Nitu, Boris Teabe, Alain Tchana, Canturk Isci, and Daniel Hagimont. Welcome to zombieland: practical and energy-efficient memory disaggregation in a datacenter. In Proceedings of the Thirteenth EuroSys Conference, EuroSys ’18, Porto, Portugal, 2018. Association for Computing Machinery.

[54] Nvidia. Rdma aware networks programming user manual, 2025. https://docs.nvidia.com/ rdma-aware-networks-programming-user-manual-1-7. pdf.

[55] John Ousterhout, Arjun Gopalan, Ashish Gupta, Ankita Kejriwal, Collin Lee, Behnam Montazeri, Diego Ongaro, Seo Jin Park, Henry Qin, Mendel Rosenblum, Stephen Rumble, Ryan Stutsman, and Stephen Yang. The ramcloud storage system. ACM Trans. Comput. Syst., 33(3), August 2015.

[56] Yifan Qiao, Chenxi Wang, Zhenyuan Ruan, Adam Belay, Qingda Lu, Yiying Zhang, Miryung Kim, and Guoqing Harry Xu. Hermit: Low-Latency, High-Throughput, and transparent remote memory via Feedback-Directed asynchrony. In 20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23), pages 181–198, Boston, MA, April 2023. USENIX Association.

[57] Waleed Reda, Marco Canini, Dejan Kostic, and Simon´ Peter. RDMA is turing complete, we just did not know it yet! In 19th USENIX Symposium on Networked Systems Design and Implementation (NSDI 22), pages 71–85, Renton, WA, April 2022. USENIX Association.

[58] Feng Ren, Mingxing Zhang, Kang Chen, Huaxia Xia, Zuoning Chen, and Yongwei Wu. Scaling up memory disaggregated applications with smart. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Oper ating Systems, Volume 1, ASPLOS ’24, page 351–367, La Jolla, CA, USA, 2024. Association for Computing Machinery.

[59] Zhenyuan Ruan, Malte Schwarzkopf, Marcos K. Aguil era, and Adam Belay. AIFM: High-Performance, Application-Integrated far memory. In 14th USENIX Symposium on Operating Systems Design and Imple mentation (OSDI 20), pages 315–332. USENIX Association, November 2020.

[60] Yizhou Shan, Yutong Huang, Yilun Chen, and Yiying Zhang. LegoOS: A disseminated, distributed OS for hardware resource disaggregation. In 13th USENIX Symposium on Operating Systems Design and Imple mentation (OSDI 18), pages 69–87, Carlsbad, CA, October 2018. USENIX Association.

[61] Jiacheng Shen, Pengfei Zuo, Xuchuan Luo, Yuxin Su, Ji azhen Gu, Hao Feng, Yangfan Zhou, and Michael R. Lyu. Ditto: An elastic and adaptive memory-disaggregated caching system. In Proceedings of the 29th Sym posium on Operating Systems Principles, SOSP ’23, page 675–691, Koblenz, Germany, 2023. Association for Computing Machinery.

[62] Jiacheng Shen, Pengfei Zuo, Xuchuan Luo, Tianyi Yang, Yuxin Su, Yangfan Zhou, and Michael R. Lyu. FUSEE: A fully Memory-Disaggregated Key-Value store. In 21st USENIX Conference on File and Storage Technolo gies (FAST 23), pages 81–98, Santa Clara, CA, February 2023. USENIX Association.

[63] Yan Sun, Yifan Yuan, Zeduo Yu, Reese Kuper, Chi hun Song, Jinghan Huang, Houxiang Ji, Siddharth Agar wal, Jiaqi Lou, Ipoom Jeong, Ren Wang, Jung Ho Ahn, Tianyin Xu, and Nam Sung Kim. Demystifying cxl

memory with genuine cxl-ready systems and devices. In Proceedings ofthe 56th Annual IEEE/ACM International Symposium on Microarchitecture, MICRO ’23, page 105–121, Toronto, ON, Canada, 2023. Association for Computing Machinery.

[64] Muhammad Tirmazi, Adam Barker, Nan Deng, Md E. Haque, Zhijing Gene Qin, Steven Hand, Mor Harchol-Balter, and John Wilkes. Borg: the next generation. In Proceedings of the Fifteenth European Conference on Computer Systems, EuroSys ’20, Heraklion, Greece, 2020. Association for Computing Machinery.

[65] Shin-Yeh Tsai, Yizhou Shan, and Yiying Zhang. Disaggregating persistent memory and controlling them remotely: An exploration of passive disaggregated Key-Value stores. In 2020 USENIX Annual Technical Conference (USENIX ATC 20), pages 33–48. USENIX Association, July 2020.

[66] Shin-Yeh Tsai and Yiying Zhang. Lite kernel rdma support for datacenter applications. In Proceedings of the 26th Symposium on Operating Systems Principles, SOSP ’17, page 306–324, Shanghai, China, 2017. Association for Computing Machinery.

[67] Alexandre Verbitski, Anurag Gupta, Debanjan Saha, Murali Brahmadesam, Kamal Gupta, Raman Mittal, Sailesh Krishnamurthy, Sandor Maurice, Tengiz Kharatishvili, and Xiaofeng Bao. Amazon aurora: Design considerations for high throughput cloud-native relational databases. In Proceedings of the 2017 ACM International Conference on Management ofData, SIGMOD ’17, page 1041–1052, Chicago, Illinois, USA, 2017. Association for Computing Machinery.

[68] Chenxi Wang, Haoran Ma, Shi Liu, Yuanqi Li, Zhenyuan Ruan, Khanh Nguyen, Michael D. Bond, Ravi Netravali, Miryung Kim, and Guoqing Harry Xu. Semeru: A Memory-Disaggregated managed runtime. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20), pages 261–280. USENIX Association, November 2020.

[69] Chenxi Wang, Haoran Ma, Shi Liu, Yifan Qiao, Jonathan Eyolfson, Christian Navasca, Shan Lu, and Guoqing Harry Xu. MemLiner: Lining up tracing and application for a Far-Memory-Friendly runtime. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), pages 35–53, Carlsbad, CA, July 2022. USENIX Association.

[70] Chenxi Wang, Yifan Qiao, Haoran Ma, Shi Liu, Wenguang Chen, Ravi Netravali, Miryung Kim, and Guoqing Harry Xu. Canvas: Isolated and adaptive swapping for Multi-Applications on remote memory. In 20th USENIX Symposium on Networked Systems Design and

Implementation (NSDI 23), pages 161–179, Boston, MA, April 2023. USENIX Association.

[71] Qing Wang, Youyou Lu, and Jiwu Shu. Sherman: A write-optimized distributed b+tree index on disaggre gated memory. In Proceedings of the 2022 International Conference on Management ofData, SIGMOD ’22, page 1033–1048, Philadelphia, PA, USA, 2022. As sociation for Computing Machinery.

[72] Qing Wang, Youyou Lu, Erci Xu, Junru Li, Youmin Chen, and Jiwu Shu. Concordia: Distributed shared memory with In-Network cache coherence. In 19th USENIX Conference on File and Storage Technologies (FAST 21), pages 277–292. USENIX Association, February 2021.

[73] Xiaoyang Wang, Yongkun Li, Kan Wu, Wenzhe Zhu, Yuqi Li, and Yinlong Xu. Finemem: Breaking the allocation overhead vs. memory waste dilemma in finegrained disaggregated memory management. In 19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25), pages 57–74, Santa Clara, CA, July 2025. USENIX Association.

[74] Xingda Wei, Zhiyuan Dong, Rong Chen, and Haibo Chen. Deconstructing RDMA-enabled distributed transactions: Hybrid is better! In 13th USENIX Sympo sium on Operating Systems Design and Implementation (OSDI 18), pages 233–251, Carlsbad, CA, October 2018. USENIX Association.

[75] Xingda Wei, Fangming Lu, Rong Chen, and Haibo Chen. KRCORE: A microsecond-scale RDMA control plane for elastic computing. In 2022 USENIX Annual Technical Conference (USENIX ATC 22), pages 121–136, Carlsbad, CA, July 2022. USENIX Association.

[76] Wikipedia. Directory-based cache coherence, 2025. https://en.wikipedia.org/wiki/ Directory-based\_cache\_coherence.

[77] Dong Xu, Yuan Feng, Kwangsik Shin, Daewoo Kim, Hyeran Jeon, and Dong Li. Efficient tensor offloading for large deep-learning model training based on compute express link. In Proceedings of the International Conferencefor High Performance Computing, Network ing, Storage, and Analysis, SC ’24, Atlanta, GA, USA, 2024. IEEE Press.

[78] Juncheng Yang, Yao Yue, and K. V. Rashmi. A large scale analysis of hundreds of in-memory cache clusters at twitter. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20), pages 191–208. USENIX Association, November 2020.

[79] Xinjun Yang, Yingqiang Zhang, Hao Chen, Feifei Li, Gerry Fan, Yang Kong, Bo Wang, Jing Fang, Yuhui Wang, Tao Huang, Wenpu Hu, Jim Kao, and Jianping Jiang. Unlocking the potential of cxl for disaggregated memory in cloud-native databases. In Companion ofthe 2025 International Conference on Management of Data, SIGMOD/PODS ’25, page 689–702, Berlin, Germany, 2025. Association for Computing Machinery.

[80] Xinjun Yang, Yingqiang Zhang, Hao Chen, Feifei Li, Bo Wang, Jing Fang, Chuan Sun, and Yuhui Wang. Polardb-mp: A multi-primary cloud-native database via disaggregated shared memory. In Companion of the 2024 International Conference on Management of Data, SIGMOD ’24, page 295–308, Santiago AA, Chile, 2024. Association for Computing Machinery.

[81] Wonsup Yoon, Jisu Ok, Jinyoung Oh, Sue Moon, and Youngjin Kwon. Dilos: Do not trade compatibility for performance in memory disaggregation. In Proceedings of the Eighteenth European Conference on Computer Systems, EuroSys ’23, page 266–282, Rome, Italy, 2023. Association for Computing Machinery.

[82] Hanze Zhang, Kaiming Wang, Rong Chen, Xingda Wei, and Haibo Chen. Difache: Efficient and scalable caching on disaggregated memory using decentralized coherence. ArXiv, abs/2505.18013, 2025.

[83] Ming Zhang, Yu Hua, and Zhijun Yang. Motor: Enabling Multi-Versioning for distributed transactions on disaggregated memory. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 801–819, Santa Clara, CA, July 2024. USENIX Association.

[84] Ming Zhang, Yu Hua, Pengfei Zuo, and Lurong Liu. FORD: Fast one-sided RDMA-based distributed transactions for disaggregated persistent memory. In 20th USENIX Conference on File and Storage Technologies (FAST 22), pages 51–68, Santa Clara, CA, February 2022. USENIX Association.

[85] Mingxing Zhang, Teng Ma, Jinqi Hua, Zheng Liu, Kang Chen, Ning Ding, Fan Du, Jinlei Jiang, Tao Ma, and Yongwei Wu. Partial failure resilient memory management system for (cxl-based) distributed shared memory. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP ’23, page 658–674, Koblenz, Germany, 2023. Association for Computing Machinery.

[86] Yuhong Zhong, Daniel S. Berger, Pantea Zardoshti, Enrique Saurez, Jacob Nelson, Dan R. K. Ports, Antonis Psistakis, Joshua Fried, and Asaf Cidon. Oasis: Pooling pcie devices over cxl to boost utilization. In Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles, SOSP ’25, page 101–119, Lotte Hotel

World, Seoul, Republic of Korea, 2025. Association for Computing Machinery.

[87] Yang Zhou, Hassan M. G. Wassel, Sihang Liu, Jiaqi Gao, James Mickens, Minlan Yu, Chris Kennelly, Paul Turner, David E. Culler, Henry M. Levy, and Amin Vahdat. Carbink: Fault-Tolerant far memory. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), pages 55–71, Carlsbad, CA, July 2022. USENIX Association.

[88] Tobias Ziegler, Jacob Nelson-Slivon, Viktor Leis, and Carsten Binnig. Design guidelines for correct, efficient, and scalable synchronization using one-sided rdma. Proc. ACM Manag. Data, 1(2), June 2023.

[89] Pengfei Zuo, Jiazhao Sun, Liu Yang, Shuangwu Zhang, and Yu Hua. One-sided RDMA-Conscious extendible hashing for disaggregated memory. In 2021 USENIX Annual Technical Conference (USENIX ATC 21), pages 15–29. USENIX Association, July 2021.