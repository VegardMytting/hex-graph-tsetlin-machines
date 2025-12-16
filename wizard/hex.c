// Copyright (c) 2024 Ole-Christoffer Granmo
// Modified for CSV dataset generation

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir(path, 0755)
#endif

#ifndef BOARD_DIM
#define BOARD_DIM 5
#endif

#ifndef NUM_GAMES
#define NUM_GAMES 1000000
#endif

int neighbors[] = { -(BOARD_DIM+2) + 1, -(BOARD_DIM+2), -1, 1, (BOARD_DIM+2), (BOARD_DIM+2) - 1 };

struct hex_game {
  int board[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
  int open_positions[BOARD_DIM*BOARD_DIM];
  int number_of_open_positions;
  int moves[BOARD_DIM*BOARD_DIM];
  int connected[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
};

void hg_init(struct hex_game *hg) {
  for (int i = 0; i < BOARD_DIM+2; ++i) {
    for (int j = 0; j < BOARD_DIM+2; ++j) {
      int idx = (i*(BOARD_DIM+2) + j) * 2;
      hg->board[idx] = 0;
      hg->board[idx + 1] = 0;

      if (i > 0 && i < BOARD_DIM + 1 &&
        j > 0 && j < BOARD_DIM + 1) {
        hg->open_positions[(i-1)*BOARD_DIM + (j-1)] =
          i*(BOARD_DIM+2) + j;
      }

      hg->connected[idx]     = (i == 0);
      hg->connected[idx + 1] = (j == 0);
    }
  }
  hg->number_of_open_positions = BOARD_DIM * BOARD_DIM;
}

int hg_connect(struct hex_game *hg, int player, int position) {
  hg->connected[position*2 + player] = 1;
  
  if (player == 0 && position / (BOARD_DIM + 2) == BOARD_DIM)
    return 1;
  if (player == 1 && position % (BOARD_DIM + 2) == BOARD_DIM)
    return 1;

  for (int i = 0; i < 6; ++i) {
    int n = position + neighbors[i];
    if (hg->board[n*2 + player] &&
      !hg->connected[n*2 + player]) {
      if (hg_connect(hg, player, n))
        return 1;
    }
  }
  return 0;
}

int hg_winner(struct hex_game *hg, int player, int position) {
  for (int i = 0; i < 6; ++i) {
    int n = position + neighbors[i];
    if (hg->connected[n*2 + player])
      return hg_connect(hg, player, position);
  }
  return 0;
}

int hg_place_piece_randomly(struct hex_game *hg, int player) {
  int idx = rand() % hg->number_of_open_positions;
  int pos = hg->open_positions[idx];

  hg->board[pos*2 + player] = 1;
  hg->moves[BOARD_DIM*BOARD_DIM - hg->number_of_open_positions] = pos;
  hg->open_positions[idx] = hg->open_positions[hg->number_of_open_positions - 1];
  hg->number_of_open_positions--;

  return pos;
}

void write_csv_header(FILE *f) {
  for (int i = 0; i < BOARD_DIM; ++i)
    for (int j = 0; j < BOARD_DIM; ++j)
      fprintf(f, "cell%d_%d,", i, j);

  fprintf(f, "winner\n");
}

void reconstruct_board(int board[], int moves[], int keep_moves) {
  memset(board, 0, sizeof(int)*(BOARD_DIM+2)*(BOARD_DIM+2)*2);

  for (int m = 0; m < keep_moves; ++m) {
    int player = m % 2;
    int pos = moves[m];
    board[pos*2 + player] = 1;
  }
}

void write_board_csv(FILE *f, int board[], int winner) {
  for (int i = 1; i <= BOARD_DIM; ++i) {
    for (int j = 1; j <= BOARD_DIM; ++j) {
      int idx = (i*(BOARD_DIM+2) + j) * 2;
      int val = 0;

      if (board[idx] == 1) val = 1;   // X
      else if (board[idx+1] == 1) val = -1; // O

      fprintf(f, "%d,", val);
    }
  }
  fprintf(f, "%d\n", winner);
}

int main(void) {
  srand((unsigned)time(NULL));

  MKDIR("data");

  char file_final[256];
  char file_m2[256];
  char file_m5[256];

  snprintf(file_final, sizeof(file_final), "data/hex_%dx%d_%d_final.csv", BOARD_DIM, BOARD_DIM, NUM_GAMES);
  snprintf(file_m2, sizeof(file_m2), "data/hex_%dx%d_%d_minus2.csv", BOARD_DIM, BOARD_DIM, NUM_GAMES);
  snprintf(file_m5, sizeof(file_m5), "data/hex_%dx%d_%d_minus5.csv", BOARD_DIM, BOARD_DIM, NUM_GAMES);

  FILE *f_final = fopen(file_final, "w");
  FILE *f_m2 = fopen(file_m2, "w");
  FILE *f_m5 = fopen(file_m5, "w");

  if (!f_final || !f_m2 || !f_m5) {
    perror("Failed to open output files");
    return 1;
  }

  write_csv_header(f_final);
  write_csv_header(f_m2);
  write_csv_header(f_m5);

  struct hex_game hg;
  int temp_board[(BOARD_DIM+2)*(BOARD_DIM+2)*2];

  printf("Generating %d games for %dx%d board...\n", NUM_GAMES, BOARD_DIM, BOARD_DIM);

  for (int g = 0; g < NUM_GAMES; ++g) {
    if (g % 10000 == 0) {
      printf("Generated %d/%d games (%.1f%%)\n", g, NUM_GAMES, 100.0*g/NUM_GAMES);
    }

    hg_init(&hg);

    int winner = 0;
    int player = 0;
    int total_moves = 0;

    while (hg.number_of_open_positions > 0) {
      int pos = hg_place_piece_randomly(&hg, player);
      total_moves++;

      if (hg_winner(&hg, player, pos)) {
        winner = (player == 0) ? 1 : -1;
        break;
      }

      player = 1 - player;
    }

    reconstruct_board(temp_board, hg.moves, total_moves);
    write_board_csv(f_final, temp_board, winner);

    if (total_moves >= 2) {
      reconstruct_board(temp_board, hg.moves, total_moves - 2);
      write_board_csv(f_m2, temp_board, winner);
    }

    if (total_moves >= 5) {
      reconstruct_board(temp_board, hg.moves, total_moves - 5);
      write_board_csv(f_m5, temp_board, winner);
    }
  }

  fclose(f_final);
  fclose(f_m2);
  fclose(f_m5);

  printf("\n✓ Successfully generated all datasets!\n");
  printf("Files created in data/ directory:\n");
  printf("  - %s\n", file_final);
  printf("  - %s\n", file_m2);
  printf("  - %s\n", file_m5);

  return 0;
}
