#pragma once

using namespace metal;

constant uint WEAVER_FRONTIER_LANE_TOKEN = 0;
constant uint WEAVER_FRONTIER_LANE_PARENT = 1;
constant uint WEAVER_FRONTIER_LANE_DEPTH = 2;
constant uint WEAVER_FRONTIER_LANE_CUM = 3;
constant uint WEAVER_FRONTIER_LANE_LOGPROB = 4;
constant uint WEAVER_FRONTIER_LANE_KEY = 5;
constant uint WEAVER_FRONTIER_LANE_ACTIVE = 6;

constant uint WEAVER_TREE_LANE_TOKEN = 0;
constant uint WEAVER_TREE_LANE_PARENT = 1;
constant uint WEAVER_TREE_LANE_DEPTH = 2;
constant uint WEAVER_TREE_LANE_CUM = 3;
constant uint WEAVER_TREE_LANE_LOGPROB = 4;
constant uint WEAVER_TREE_LANE_MASK = 5;

constant uint WEAVER_METADATA_LANE_DEPTH = 0;
constant uint WEAVER_METADATA_LANE_ANCESTOR_COUNT = 1;
constant uint WEAVER_METADATA_LANE_NODE_INDEX = 2;

constant uint WEAVER_FRONTIER_NO_WINNER = 0xffffffffu;
constant uint WEAVER_FRONTIER_SELECT_THREADS = 256;
constant uint WEAVER_FRONTIER_SELECT_SIMDGROUPS = 8;
constant uint WEAVER_FRONTIER_ENTRIES_PER_THREAD = 8;
constant uint WEAVER_FRONTIER_MAX_SLOTS = 2048;
constant uint WEAVER_FRONTIER_MAX_WIDTH = 32;
