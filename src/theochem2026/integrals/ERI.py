import numpy as np
from numba import njit

ERI_KEY_TO_CASE = {
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): 0,
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1): 1,
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0): 2,
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0): 3,
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0): 4,
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1): 5,
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0): 6,
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0): 7,
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0): 8,
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1): 9,
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0): 10,
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0): 11,
    (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0): 12,
    (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1): 13,
    (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0): 14,
    (0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0): 15,
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0): 16,
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1): 17,
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0): 18,
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0): 19,
    (0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0): 20,
    (0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1): 21,
    (0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0): 22,
    (0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0): 23,
    (0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0): 24,
    (0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1): 25,
    (0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0): 26,
    (0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0): 27,
    (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0): 28,
    (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1): 29,
    (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0): 30,
    (0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0): 31,
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0): 32,
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1): 33,
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0): 34,
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0): 35,
    (0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0): 36,
    (0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1): 37,
    (0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0): 38,
    (0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0): 39,
    (0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0): 40,
    (0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1): 41,
    (0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0): 42,
    (0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0): 43,
    (0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0): 44,
    (0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1): 45,
    (0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0): 46,
    (0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0): 47,
    (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0): 48,
    (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1): 49,
    (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0): 50,
    (0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0): 51,
    (0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0): 52,
    (0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1): 53,
    (0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0): 54,
    (0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0): 55,
    (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0): 56,
    (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1): 57,
    (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0): 58,
    (0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0): 59,
    (0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0): 60,
    (0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1): 61,
    (0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0): 62,
    (0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0): 63,
    (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0): 64,
    (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1): 65,
    (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0): 66,
    (0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0): 67,
    (0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0): 68,
    (0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1): 69,
    (0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0): 70,
    (0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0): 71,
    (0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0): 72,
    (0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1): 73,
    (0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0): 74,
    (0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0): 75,
    (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0): 76,
    (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1): 77,
    (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0): 78,
    (0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0): 79,
    (0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0): 80,
    (0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1): 81,
    (0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0): 82,
    (0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0): 83,
    (0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0): 84,
    (0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1): 85,
    (0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0): 86,
    (0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0): 87,
    (0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0): 88,
    (0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1): 89,
    (0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0): 90,
    (0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0): 91,
    (0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0): 92,
    (0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1): 93,
    (0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0): 94,
    (0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0): 95,
    (0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0): 96,
    (0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1): 97,
    (0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0): 98,
    (0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0): 99,
    (0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0): 100,
    (0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1): 101,
    (0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0): 102,
    (0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0): 103,
    (0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0): 104,
    (0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1): 105,
    (0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0): 106,
    (0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0): 107,
    (0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0): 108,
    (0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1): 109,
    (0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0): 110,
    (0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0): 111,
    (0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0): 112,
    (0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1): 113,
    (0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0): 114,
    (0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0): 115,
    (0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0): 116,
    (0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1): 117,
    (0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0): 118,
    (0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0): 119,
    (0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0): 120,
    (0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1): 121,
    (0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0): 122,
    (0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0): 123,
    (0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0): 124,
    (0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1): 125,
    (0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0): 126,
    (0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0): 127,
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): 128,
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1): 129,
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0): 130,
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0): 131,
    (0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0): 132,
    (0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1): 133,
    (0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0): 134,
    (0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0): 135,
    (0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0): 136,
    (0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1): 137,
    (0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0): 138,
    (0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0): 139,
    (0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0): 140,
    (0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1): 141,
    (0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0): 142,
    (0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0): 143,
    (0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0): 144,
    (0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1): 145,
    (0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0): 146,
    (0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0): 147,
    (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0): 148,
    (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1): 149,
    (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0): 150,
    (0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0): 151,
    (0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0): 152,
    (0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1): 153,
    (0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0): 154,
    (0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0): 155,
    (0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0): 156,
    (0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1): 157,
    (0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0): 158,
    (0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0): 159,
    (0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0): 160,
    (0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1): 161,
    (0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0): 162,
    (0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0): 163,
    (0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0): 164,
    (0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1): 165,
    (0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0): 166,
    (0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0): 167,
    (0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0): 168,
    (0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1): 169,
    (0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0): 170,
    (0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0): 171,
    (0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0): 172,
    (0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1): 173,
    (0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0): 174,
    (0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0): 175,
    (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0): 176,
    (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1): 177,
    (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0): 178,
    (0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0): 179,
    (0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0): 180,
    (0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1): 181,
    (0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0): 182,
    (0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0): 183,
    (0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0): 184,
    (0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1): 185,
    (0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0): 186,
    (0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0): 187,
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0): 188,
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1): 189,
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0): 190,
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0): 191,
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): 192,
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1): 193,
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0): 194,
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0): 195,
    (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0): 196,
    (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1): 197,
    (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0): 198,
    (1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0): 199,
    (1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0): 200,
    (1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1): 201,
    (1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0): 202,
    (1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0): 203,
    (1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0): 204,
    (1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1): 205,
    (1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0): 206,
    (1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0): 207,
    (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0): 208,
    (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1): 209,
    (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0): 210,
    (1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0): 211,
    (1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0): 212,
    (1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1): 213,
    (1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0): 214,
    (1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0): 215,
    (1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0): 216,
    (1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1): 217,
    (1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0): 218,
    (1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0): 219,
    (1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0): 220,
    (1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1): 221,
    (1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0): 222,
    (1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0): 223,
    (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0): 224,
    (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1): 225,
    (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0): 226,
    (1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0): 227,
    (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0): 228,
    (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1): 229,
    (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0): 230,
    (1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0): 231,
    (1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0): 232,
    (1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1): 233,
    (1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0): 234,
    (1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0): 235,
    (1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0): 236,
    (1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1): 237,
    (1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0): 238,
    (1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0): 239,
    (1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0): 240,
    (1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1): 241,
    (1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0): 242,
    (1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0): 243,
    (1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0): 244,
    (1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1): 245,
    (1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0): 246,
    (1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0): 247,
    (1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0): 248,
    (1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1): 249,
    (1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0): 250,
    (1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0): 251,
    (1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0): 252,
    (1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1): 253,
    (1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0): 254,
    (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0): 255,
}

@njit(cache=True, fastmath=True)
def ERI(case_id, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, ABx, ABy, ABz, CDx, CDy, CDz, p, q, alpha, beta, gamma, delta, Px, Py, Pz, Qx, Qy, Qz, PQx, PQy, PQz, rAB, rCD, pRPQ, u, F0, F1, F2, F3, F4, KAB, KCD):
    if case_id == 0:
        t0 = p ** (-1.0)
        t1 = q ** (-1.0)
        return 2 * np.pi ** 2.5 * t0 * t1 * (p + q) ** (-0.5) * KAB * KCD * F0
    if case_id == 1:
        t0 = p ** (-1.0)
        t1 = p + q
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-2.0) * (CDz * gamma * t1 ** 1.5 * F0 - t1 ** 0.5 * (p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)) * F1) * KAB * KCD / q ** 2
    if case_id == 2:
        t0 = p ** (-1.0)
        t1 = p + q
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-2.0) * (CDy * gamma * t1 ** 1.5 * F0 - t1 ** 0.5 * (p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)) * F1) * KAB * KCD / q ** 2
    if case_id == 3:
        t0 = p ** (-1.0)
        t1 = p + q
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-2.0) * (CDx * gamma * t1 ** 1.5 * F0 - t1 ** 0.5 * (p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)) * F1) * KAB * KCD / q ** 2
    if case_id == 4:
        t0 = p ** (-1.0)
        t1 = p + q
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-2.0) * (-CDz * delta * t1 ** 1.5 * F0 + t1 ** 0.5 * (-p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)) * F1) * KAB * KCD / q ** 2
    if case_id == 5:
        t0 = p ** (-1.0)
        t1 = p + q
        t2 = delta * gamma
        t3 = F0
        t4 = t1 ** 6.0
        t5 = t1 ** 5.0 * F1
        t6 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        return np.pi ** 2.5 * t0 * t1 ** (-6.5) * (-2 * CDz ** 2 * t2 * t3 * t4 - 2 * CDz * t5 * t6 * (delta - gamma) - p * q * t5 + q * t3 * t4 + 2 * t1 ** 4.0 * t6 ** 2 * F2) * KAB * KCD / q ** 3
    if case_id == 6:
        t0 = p ** (-1.0)
        t1 = p + q
        t2 = CDy * gamma
        t3 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t4 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-4.5) * (-CDz * delta * t1 ** 4.0 * t2 * F0 + t1 ** 2.0 * t3 * t4 * F2 - t1 ** 3.0 * (CDz * delta * t3 - t2 * t4) * F1) * KAB * KCD / q ** 3
    if case_id == 7:
        t0 = p ** (-1.0)
        t1 = p + q
        t2 = CDx * gamma
        t3 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t4 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-4.5) * (-CDz * delta * t1 ** 4.0 * t2 * F0 + t1 ** 2.0 * t3 * t4 * F2 - t1 ** 3.0 * (CDz * delta * t3 - t2 * t4) * F1) * KAB * KCD / q ** 3
    if case_id == 8:
        t0 = p ** (-1.0)
        t1 = p + q
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-2.0) * (-CDy * delta * t1 ** 1.5 * F0 + t1 ** 0.5 * (-p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)) * F1) * KAB * KCD / q ** 2
    if case_id == 9:
        t0 = p ** (-1.0)
        t1 = p + q
        t2 = CDy * delta
        t3 = CDz * gamma
        t4 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t5 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-4.5) * (t1 ** 2.0 * t4 * t5 * F2 - t1 ** 3.0 * (t2 * t5 - t3 * t4) * F1 - t1 ** 4.0 * t2 * t3 * F0) * KAB * KCD / q ** 3
    if case_id == 10:
        t0 = p ** (-1.0)
        t1 = p + q
        t2 = delta * gamma
        t3 = F0
        t4 = t1 ** 6.0
        t5 = t1 ** 5.0 * F1
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        return np.pi ** 2.5 * t0 * t1 ** (-6.5) * (-2 * CDy ** 2 * t2 * t3 * t4 - 2 * CDy * t5 * t6 * (delta - gamma) - p * q * t5 + q * t3 * t4 + 2 * t1 ** 4.0 * t6 ** 2 * F2) * KAB * KCD / q ** 3
    if case_id == 11:
        t0 = p ** (-1.0)
        t1 = p + q
        t2 = CDx * gamma
        t3 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t4 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-4.5) * (-CDy * delta * t1 ** 4.0 * t2 * F0 + t1 ** 2.0 * t3 * t4 * F2 - t1 ** 3.0 * (CDy * delta * t3 - t2 * t4) * F1) * KAB * KCD / q ** 3
    if case_id == 12:
        t0 = p ** (-1.0)
        t1 = p + q
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-2.0) * (-CDx * delta * t1 ** 1.5 * F0 + t1 ** 0.5 * (-p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)) * F1) * KAB * KCD / q ** 2
    if case_id == 13:
        t0 = p ** (-1.0)
        t1 = p + q
        t2 = CDx * delta
        t3 = CDz * gamma
        t4 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t5 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-4.5) * (t1 ** 2.0 * t4 * t5 * F2 - t1 ** 3.0 * (t2 * t5 - t3 * t4) * F1 - t1 ** 4.0 * t2 * t3 * F0) * KAB * KCD / q ** 3
    if case_id == 14:
        t0 = p ** (-1.0)
        t1 = p + q
        t2 = CDx * delta
        t3 = CDy * gamma
        t4 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t5 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-4.5) * (t1 ** 2.0 * t4 * t5 * F2 - t1 ** 3.0 * (t2 * t5 - t3 * t4) * F1 - t1 ** 4.0 * t2 * t3 * F0) * KAB * KCD / q ** 3
    if case_id == 15:
        t0 = p ** (-1.0)
        t1 = p + q
        t2 = delta * gamma
        t3 = F0
        t4 = t1 ** 6.0
        t5 = t1 ** 5.0 * F1
        t6 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        return np.pi ** 2.5 * t0 * t1 ** (-6.5) * (-2 * CDx ** 2 * t2 * t3 * t4 - 2 * CDx * t5 * t6 * (delta - gamma) - p * q * t5 + q * t3 * t4 + 2 * t1 ** 4.0 * t6 ** 2 * F2) * KAB * KCD / q ** 3
    if case_id == 16:
        t0 = q ** (-1.0)
        t1 = p + q
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-2.0) * (ABz * alpha * t1 ** 1.5 * F0 - t1 ** 0.5 * (-p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)) * F1) * KAB * KCD / p ** 2
    if case_id == 17:
        t0 = p + q
        t1 = t0 ** 4.5 * F1
        t2 = ABz * alpha
        t3 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        return np.pi ** 2.5 * t0 ** (-6.0) * (2 * CDz * gamma * t0 ** 5.5 * t2 * F0 + p * q * t1 - 2 * t0 ** 3.5 * t3 ** 2 * F2 + 2 * t1 * t3 * (CDz * gamma - t2)) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 18:
        t0 = p + q
        t1 = ABz * alpha
        t2 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t3 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (CDy * gamma * t0 ** 4.0 * t1 * F0 - t0 ** 2.0 * t2 * t3 * F2 + t0 ** 3.0 * (CDy * gamma * t3 - t1 * t2) * F1) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 19:
        t0 = p + q
        t1 = ABz * alpha
        t2 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t3 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (CDx * gamma * t0 ** 4.0 * t1 * F0 - t0 ** 2.0 * t2 * t3 * F2 + t0 ** 3.0 * (CDx * gamma * t3 - t1 * t2) * F1) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 20:
        t0 = p + q
        t1 = F1
        t2 = t0 ** 4.5
        t3 = ABz * alpha
        t4 = CDz * delta
        t5 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        return np.pi ** 2.5 * t0 ** (-6.0) * (p * q * t1 * t2 - 2 * t0 ** 3.5 * t5 ** 2 * F2 - 2 * t0 ** 5.5 * t3 * t4 * F0 - 2 * t1 * t2 * t5 * (t3 + t4)) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 21:
        t0 = p + q
        t1 = F0
        t2 = t0 ** 13.5
        t3 = ABz * alpha
        t4 = CDz * delta - CDz * gamma + t3
        t5 = t0 ** 12.5 * F1
        t6 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t7 = F2
        t8 = t0 ** 11.5
        return np.pi ** 2.5 * t0 ** (-14.0) * (ABz * alpha * q * t1 * t2 - 2 * CDz ** 2 * delta * gamma * t1 * t2 * t3 - 2 * CDz * t5 * t6 * (ABz * alpha * delta - CDz * delta * gamma - gamma * t3) - p * q * t4 * t5 + 3 * p * q * t6 * t7 * t8 - q * t5 * t6 - 2 * t0 ** 10.5 * t6 ** 3 * F3 + 2 * t4 * t6 ** 2 * t7 * t8) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 22:
        t0 = p + q
        t1 = CDy * gamma
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABz * alpha
        t5 = CDz * delta
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (ABz * CDz * alpha * delta * t6 - t4 * t9 - t5 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (t4 * t6 + t5 * t6 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 23:
        t0 = p + q
        t1 = CDx * gamma
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABz * alpha
        t5 = CDz * delta
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (ABz * CDz * alpha * delta * t6 - t4 * t9 - t5 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (t4 * t6 + t5 * t6 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 24:
        t0 = p + q
        t1 = ABz * alpha
        t2 = CDy * delta
        t3 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t4 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (-t0 ** 2.0 * t3 * t4 * F2 + t0 ** 3.0 * (t1 * t3 + t2 * t4) * F1 - t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 25:
        t0 = p + q
        t1 = CDy * delta
        t2 = t0 ** 10.5 * F1
        t3 = CDz * gamma
        t4 = ABz * alpha
        t5 = F2
        t6 = t0 ** 9.5
        t7 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t8 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (-p * q * t1 * t2 + p * q * t5 * t6 * t7 - 2 * t0 ** 8.5 * t7 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t3 * t4 * F0 - 2 * t2 * (ABz * CDy * alpha * delta * t8 - t3 * t4 * t7 - t3 * t9) - 2 * t5 * t6 * t8 * (CDz * gamma * t7 - t4 * t7 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 26:
        t0 = p + q
        t1 = delta * gamma
        t2 = F0
        t3 = t0 ** 13.5
        t4 = t0 ** 12.5 * F1
        t5 = ABz * alpha
        t6 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t7 = F2
        t8 = t0 ** 11.5
        t9 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t10 = t5 * t9
        t11 = CDy * t6
        return np.pi ** 2.5 * t0 ** (-14.0) * (ABz * alpha * q * t2 * t3 - 2 * CDy ** 2 * t1 * t2 * t3 * t5 - 2 * CDy * t4 * (ABz * alpha * delta * t9 - delta * gamma * t11 - gamma * t10) - p * q * t4 * t5 + p * q * t6 * t7 * t8 - q * t4 * t6 - 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - 2 * t7 * t8 * t9 * (CDy * gamma * t6 - delta * t11 - t10)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 27:
        t0 = p + q
        t1 = CDx * gamma
        t2 = ABz * alpha
        t3 = CDy * delta
        t4 = t2 * t3
        t5 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t8 = -t6
        t9 = t2 * t8
        t10 = -t7
        t11 = t10 * t3
        t12 = -t5
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (-t1 * t10 * t8 + t11 * t12 + t12 * t9) * F2 + t0 ** 6.5 * (t1 * t11 + t1 * t9 - t12 * t4) * F1 - t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 28:
        t0 = p + q
        t1 = ABz * alpha
        t2 = CDx * delta
        t3 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t4 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (-t0 ** 2.0 * t3 * t4 * F2 + t0 ** 3.0 * (t1 * t3 + t2 * t4) * F1 - t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 29:
        t0 = p + q
        t1 = CDx * delta
        t2 = t0 ** 10.5 * F1
        t3 = CDz * gamma
        t4 = ABz * alpha
        t5 = F2
        t6 = t0 ** 9.5
        t7 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t8 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (-p * q * t1 * t2 + p * q * t5 * t6 * t7 - 2 * t0 ** 8.5 * t7 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t3 * t4 * F0 - 2 * t2 * (ABz * CDx * alpha * delta * t8 - t3 * t4 * t7 - t3 * t9) - 2 * t5 * t6 * t8 * (CDz * gamma * t7 - t4 * t7 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 30:
        t0 = p + q
        t1 = CDy * gamma
        t2 = ABz * alpha
        t3 = CDx * delta
        t4 = t2 * t3
        t5 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t8 = -t5
        t9 = t2 * t8
        t10 = -t7
        t11 = t10 * t3
        t12 = -t6
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (-t1 * t10 * t8 + t11 * t12 + t12 * t9) * F2 + t0 ** 6.5 * (t1 * t11 + t1 * t9 - t12 * t4) * F1 - t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 31:
        t0 = p + q
        t1 = delta * gamma
        t2 = F0
        t3 = t0 ** 13.5
        t4 = t0 ** 12.5 * F1
        t5 = ABz * alpha
        t6 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t7 = F2
        t8 = t0 ** 11.5
        t9 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t10 = t5 * t9
        t11 = CDx * t6
        return np.pi ** 2.5 * t0 ** (-14.0) * (ABz * alpha * q * t2 * t3 - 2 * CDx ** 2 * t1 * t2 * t3 * t5 - 2 * CDx * t4 * (ABz * alpha * delta * t9 - delta * gamma * t11 - gamma * t10) - p * q * t4 * t5 + p * q * t6 * t7 * t8 - q * t4 * t6 - 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - 2 * t7 * t8 * t9 * (CDx * gamma * t6 - delta * t11 - t10)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 32:
        t0 = q ** (-1.0)
        t1 = p + q
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-2.0) * (ABy * alpha * t1 ** 1.5 * F0 - t1 ** 0.5 * (-p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)) * F1) * KAB * KCD / p ** 2
    if case_id == 33:
        t0 = p + q
        t1 = ABy * alpha
        t2 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t3 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (CDz * gamma * t0 ** 4.0 * t1 * F0 - t0 ** 2.0 * t2 * t3 * F2 + t0 ** 3.0 * (CDz * gamma * t2 - t1 * t3) * F1) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 34:
        t0 = p + q
        t1 = t0 ** 4.5 * F1
        t2 = ABy * alpha
        t3 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        return np.pi ** 2.5 * t0 ** (-6.0) * (2 * CDy * gamma * t0 ** 5.5 * t2 * F0 + p * q * t1 - 2 * t0 ** 3.5 * t3 ** 2 * F2 + 2 * t1 * t3 * (CDy * gamma - t2)) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 35:
        t0 = p + q
        t1 = ABy * alpha
        t2 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t3 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (CDx * gamma * t0 ** 4.0 * t1 * F0 - t0 ** 2.0 * t2 * t3 * F2 + t0 ** 3.0 * (CDx * gamma * t3 - t1 * t2) * F1) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 36:
        t0 = p + q
        t1 = ABy * alpha
        t2 = CDz * delta
        t3 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t4 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (-t0 ** 2.0 * t3 * t4 * F2 + t0 ** 3.0 * (t1 * t4 + t2 * t3) * F1 - t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 37:
        t0 = p + q
        t1 = delta * gamma
        t2 = F0
        t3 = t0 ** 13.5
        t4 = t0 ** 12.5 * F1
        t5 = ABy * alpha
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = F2
        t8 = t0 ** 11.5
        t9 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t10 = t5 * t9
        t11 = CDz * t6
        return np.pi ** 2.5 * t0 ** (-14.0) * (ABy * alpha * q * t2 * t3 - 2 * CDz ** 2 * t1 * t2 * t3 * t5 - 2 * CDz * t4 * (ABy * alpha * delta * t9 - delta * gamma * t11 - gamma * t10) - p * q * t4 * t5 + p * q * t6 * t7 * t8 - q * t4 * t6 - 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - 2 * t7 * t8 * t9 * (CDz * gamma * t6 - delta * t11 - t10)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 38:
        t0 = p + q
        t1 = CDz * delta
        t2 = t0 ** 10.5 * F1
        t3 = CDy * gamma
        t4 = ABy * alpha
        t5 = F2
        t6 = t0 ** 9.5
        t7 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t8 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (-p * q * t1 * t2 + p * q * t5 * t6 * t7 - 2 * t0 ** 8.5 * t7 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t3 * t4 * F0 - 2 * t2 * (ABy * CDz * alpha * delta * t8 - t3 * t4 * t7 - t3 * t9) - 2 * t5 * t6 * t8 * (CDy * gamma * t7 - t4 * t7 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 39:
        t0 = p + q
        t1 = CDx * gamma
        t2 = ABy * alpha
        t3 = CDz * delta
        t4 = t2 * t3
        t5 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t8 = -t7
        t9 = t2 * t8
        t10 = -t6
        t11 = t10 * t3
        t12 = -t5
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (-t1 * t10 * t8 + t11 * t12 + t12 * t9) * F2 + t0 ** 6.5 * (t1 * t11 + t1 * t9 - t12 * t4) * F1 - t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 40:
        t0 = p + q
        t1 = F1
        t2 = t0 ** 4.5
        t3 = ABy * alpha
        t4 = CDy * delta
        t5 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        return np.pi ** 2.5 * t0 ** (-6.0) * (p * q * t1 * t2 - 2 * t0 ** 3.5 * t5 ** 2 * F2 - 2 * t0 ** 5.5 * t3 * t4 * F0 - 2 * t1 * t2 * t5 * (t3 + t4)) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 41:
        t0 = p + q
        t1 = CDz * gamma
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABy * alpha
        t5 = CDy * delta
        t6 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (ABy * CDy * alpha * delta * t6 - t4 * t9 - t5 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (t4 * t6 + t5 * t6 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 42:
        t0 = p + q
        t1 = F0
        t2 = t0 ** 13.5
        t3 = ABy * alpha
        t4 = CDy * delta - CDy * gamma + t3
        t5 = t0 ** 12.5 * F1
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = F2
        t8 = t0 ** 11.5
        return np.pi ** 2.5 * t0 ** (-14.0) * (ABy * alpha * q * t1 * t2 - 2 * CDy ** 2 * delta * gamma * t1 * t2 * t3 - 2 * CDy * t5 * t6 * (ABy * alpha * delta - CDy * delta * gamma - gamma * t3) - p * q * t4 * t5 + 3 * p * q * t6 * t7 * t8 - q * t5 * t6 - 2 * t0 ** 10.5 * t6 ** 3 * F3 + 2 * t4 * t6 ** 2 * t7 * t8) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 43:
        t0 = p + q
        t1 = CDx * gamma
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABy * alpha
        t5 = CDy * delta
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (ABy * CDy * alpha * delta * t6 - t4 * t9 - t5 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (t4 * t6 + t5 * t6 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 44:
        t0 = p + q
        t1 = ABy * alpha
        t2 = CDx * delta
        t3 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t4 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (-t0 ** 2.0 * t3 * t4 * F2 + t0 ** 3.0 * (t1 * t3 + t2 * t4) * F1 - t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 45:
        t0 = p + q
        t1 = CDz * gamma
        t2 = ABy * alpha
        t3 = CDx * delta
        t4 = t2 * t3
        t5 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t8 = -t5
        t9 = t2 * t8
        t10 = -t6
        t11 = t10 * t3
        t12 = -t7
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (-t1 * t10 * t8 + t11 * t12 + t12 * t9) * F2 + t0 ** 6.5 * (t1 * t11 + t1 * t9 - t12 * t4) * F1 - t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 46:
        t0 = p + q
        t1 = CDx * delta
        t2 = t0 ** 10.5 * F1
        t3 = CDy * gamma
        t4 = ABy * alpha
        t5 = F2
        t6 = t0 ** 9.5
        t7 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t8 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (-p * q * t1 * t2 + p * q * t5 * t6 * t7 - 2 * t0 ** 8.5 * t7 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t3 * t4 * F0 - 2 * t2 * (ABy * CDx * alpha * delta * t8 - t3 * t4 * t7 - t3 * t9) - 2 * t5 * t6 * t8 * (CDy * gamma * t7 - t4 * t7 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 47:
        t0 = p + q
        t1 = delta * gamma
        t2 = F0
        t3 = t0 ** 13.5
        t4 = t0 ** 12.5 * F1
        t5 = ABy * alpha
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = F2
        t8 = t0 ** 11.5
        t9 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t10 = t5 * t9
        t11 = CDx * t6
        return np.pi ** 2.5 * t0 ** (-14.0) * (ABy * alpha * q * t2 * t3 - 2 * CDx ** 2 * t1 * t2 * t3 * t5 - 2 * CDx * t4 * (ABy * alpha * delta * t9 - delta * gamma * t11 - gamma * t10) - p * q * t4 * t5 + p * q * t6 * t7 * t8 - q * t4 * t6 - 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - 2 * t7 * t8 * t9 * (CDx * gamma * t6 - delta * t11 - t10)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 48:
        t0 = q ** (-1.0)
        t1 = p + q
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-2.0) * (ABx * alpha * t1 ** 1.5 * F0 - t1 ** 0.5 * (-p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)) * F1) * KAB * KCD / p ** 2
    if case_id == 49:
        t0 = p + q
        t1 = ABx * alpha
        t2 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t3 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (CDz * gamma * t0 ** 4.0 * t1 * F0 - t0 ** 2.0 * t2 * t3 * F2 + t0 ** 3.0 * (CDz * gamma * t2 - t1 * t3) * F1) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 50:
        t0 = p + q
        t1 = ABx * alpha
        t2 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t3 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (CDy * gamma * t0 ** 4.0 * t1 * F0 - t0 ** 2.0 * t2 * t3 * F2 + t0 ** 3.0 * (CDy * gamma * t2 - t1 * t3) * F1) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 51:
        t0 = p + q
        t1 = t0 ** 4.5 * F1
        t2 = ABx * alpha
        t3 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        return np.pi ** 2.5 * t0 ** (-6.0) * (2 * CDx * gamma * t0 ** 5.5 * t2 * F0 + p * q * t1 - 2 * t0 ** 3.5 * t3 ** 2 * F2 + 2 * t1 * t3 * (CDx * gamma - t2)) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 52:
        t0 = p + q
        t1 = ABx * alpha
        t2 = CDz * delta
        t3 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t4 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (-t0 ** 2.0 * t3 * t4 * F2 + t0 ** 3.0 * (t1 * t4 + t2 * t3) * F1 - t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 53:
        t0 = p + q
        t1 = delta * gamma
        t2 = F0
        t3 = t0 ** 13.5
        t4 = t0 ** 12.5 * F1
        t5 = ABx * alpha
        t6 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t7 = F2
        t8 = t0 ** 11.5
        t9 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t10 = t5 * t9
        t11 = CDz * t6
        return np.pi ** 2.5 * t0 ** (-14.0) * (ABx * alpha * q * t2 * t3 - 2 * CDz ** 2 * t1 * t2 * t3 * t5 - 2 * CDz * t4 * (ABx * alpha * delta * t9 - delta * gamma * t11 - gamma * t10) - p * q * t4 * t5 + p * q * t6 * t7 * t8 - q * t4 * t6 - 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - 2 * t7 * t8 * t9 * (CDz * gamma * t6 - delta * t11 - t10)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 54:
        t0 = p + q
        t1 = CDy * gamma
        t2 = ABx * alpha
        t3 = CDz * delta
        t4 = t2 * t3
        t5 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t8 = -t7
        t9 = t2 * t8
        t10 = -t5
        t11 = t10 * t3
        t12 = -t6
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (-t1 * t10 * t8 + t11 * t12 + t12 * t9) * F2 + t0 ** 6.5 * (t1 * t11 + t1 * t9 - t12 * t4) * F1 - t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 55:
        t0 = p + q
        t1 = CDz * delta
        t2 = t0 ** 10.5 * F1
        t3 = CDx * gamma
        t4 = ABx * alpha
        t5 = F2
        t6 = t0 ** 9.5
        t7 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t8 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (-p * q * t1 * t2 + p * q * t5 * t6 * t7 - 2 * t0 ** 8.5 * t7 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t3 * t4 * F0 - 2 * t2 * (ABx * CDz * alpha * delta * t8 - t3 * t4 * t7 - t3 * t9) - 2 * t5 * t6 * t8 * (CDx * gamma * t7 - t4 * t7 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 56:
        t0 = p + q
        t1 = ABx * alpha
        t2 = CDy * delta
        t3 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t4 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (-t0 ** 2.0 * t3 * t4 * F2 + t0 ** 3.0 * (t1 * t4 + t2 * t3) * F1 - t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 57:
        t0 = p + q
        t1 = CDz * gamma
        t2 = ABx * alpha
        t3 = CDy * delta
        t4 = t2 * t3
        t5 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t8 = -t6
        t9 = t2 * t8
        t10 = -t5
        t11 = t10 * t3
        t12 = -t7
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (-t1 * t10 * t8 + t11 * t12 + t12 * t9) * F2 + t0 ** 6.5 * (t1 * t11 + t1 * t9 - t12 * t4) * F1 - t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 58:
        t0 = p + q
        t1 = delta * gamma
        t2 = F0
        t3 = t0 ** 13.5
        t4 = t0 ** 12.5 * F1
        t5 = ABx * alpha
        t6 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t7 = F2
        t8 = t0 ** 11.5
        t9 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t10 = t5 * t9
        t11 = CDy * t6
        return np.pi ** 2.5 * t0 ** (-14.0) * (ABx * alpha * q * t2 * t3 - 2 * CDy ** 2 * t1 * t2 * t3 * t5 - 2 * CDy * t4 * (ABx * alpha * delta * t9 - delta * gamma * t11 - gamma * t10) - p * q * t4 * t5 + p * q * t6 * t7 * t8 - q * t4 * t6 - 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - 2 * t7 * t8 * t9 * (CDy * gamma * t6 - delta * t11 - t10)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 59:
        t0 = p + q
        t1 = CDy * delta
        t2 = t0 ** 10.5 * F1
        t3 = CDx * gamma
        t4 = ABx * alpha
        t5 = F2
        t6 = t0 ** 9.5
        t7 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t8 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (-p * q * t1 * t2 + p * q * t5 * t6 * t7 - 2 * t0 ** 8.5 * t7 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t3 * t4 * F0 - 2 * t2 * (ABx * CDy * alpha * delta * t8 - t3 * t4 * t7 - t3 * t9) - 2 * t5 * t6 * t8 * (CDx * gamma * t7 - t4 * t7 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 60:
        t0 = p + q
        t1 = F1
        t2 = t0 ** 4.5
        t3 = ABx * alpha
        t4 = CDx * delta
        t5 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        return np.pi ** 2.5 * t0 ** (-6.0) * (p * q * t1 * t2 - 2 * t0 ** 3.5 * t5 ** 2 * F2 - 2 * t0 ** 5.5 * t3 * t4 * F0 - 2 * t1 * t2 * t5 * (t3 + t4)) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 61:
        t0 = p + q
        t1 = CDz * gamma
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABx * alpha
        t5 = CDx * delta
        t6 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (ABx * CDx * alpha * delta * t6 - t4 * t9 - t5 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (t4 * t6 + t5 * t6 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 62:
        t0 = p + q
        t1 = CDy * gamma
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABx * alpha
        t5 = CDx * delta
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (ABx * CDx * alpha * delta * t6 - t4 * t9 - t5 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (t4 * t6 + t5 * t6 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 63:
        t0 = p + q
        t1 = F0
        t2 = t0 ** 13.5
        t3 = ABx * alpha
        t4 = CDx * delta - CDx * gamma + t3
        t5 = t0 ** 12.5 * F1
        t6 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t7 = F2
        t8 = t0 ** 11.5
        return np.pi ** 2.5 * t0 ** (-14.0) * (ABx * alpha * q * t1 * t2 - 2 * CDx ** 2 * delta * gamma * t1 * t2 * t3 - 2 * CDx * t5 * t6 * (ABx * alpha * delta - CDx * delta * gamma - gamma * t3) - p * q * t4 * t5 + 3 * p * q * t6 * t7 * t8 - q * t5 * t6 - 2 * t0 ** 10.5 * t6 ** 3 * F3 + 2 * t4 * t6 ** 2 * t7 * t8) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 64:
        t0 = q ** (-1.0)
        t1 = p + q
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-2.0) * (-ABz * beta * t1 ** 1.5 * F0 + t1 ** 0.5 * (p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)) * F1) * KAB * KCD / p ** 2
    if case_id == 65:
        t0 = p + q
        t1 = F1
        t2 = t0 ** 4.5
        t3 = ABz * beta
        t4 = CDz * gamma
        t5 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        return np.pi ** 2.5 * t0 ** (-6.0) * (p * q * t1 * t2 - 2 * t0 ** 3.5 * t5 ** 2 * F2 - 2 * t0 ** 5.5 * t3 * t4 * F0 - 2 * t1 * t2 * t5 * (t3 + t4)) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 66:
        t0 = p + q
        t1 = ABz * beta
        t2 = CDy * gamma
        t3 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t4 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        return -2 * np.pi ** 2.5 * t0 ** (-4.5) * (t0 ** 2.0 * t3 * t4 * F2 - t0 ** 3.0 * (t1 * t3 + t2 * t4) * F1 + t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 67:
        t0 = p + q
        t1 = ABz * beta
        t2 = CDx * gamma
        t3 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t4 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        return -2 * np.pi ** 2.5 * t0 ** (-4.5) * (t0 ** 2.0 * t3 * t4 * F2 - t0 ** 3.0 * (t1 * t3 + t2 * t4) * F1 + t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 68:
        t0 = p + q
        t1 = t0 ** 4.5 * F1
        t2 = ABz * beta
        t3 = CDz * delta
        t4 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        return np.pi ** 2.5 * t0 ** (-6.0) * (p * q * t1 - 2 * t0 ** 3.5 * t4 ** 2 * F2 + 2 * t0 ** 5.5 * t2 * t3 * F0 + 2 * t1 * t4 * (t2 - t3)) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 69:
        t0 = p + q
        t1 = ABz * beta
        t2 = t0 ** 13.5 * F0
        t3 = -CDz * delta + CDz * gamma + t1
        t4 = t0 ** 12.5 * F1
        t5 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t6 = q * t5
        t7 = t0 ** 11.5 * F2
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * CDz ** 2 * delta * gamma * t1 * t2 + 2 * CDz * t4 * t5 * (ABz * beta * gamma - CDz * delta * gamma - delta * t1) + p * q * t3 * t4 - 3 * p * t6 * t7 - q * t1 * t2 + 2 * t0 ** 10.5 * t5 ** 3 * F3 - 2 * t3 * t5 ** 2 * t7 + t4 * t6) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 70:
        t0 = p + q
        t1 = CDy * gamma
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = CDz * delta
        t5 = ABz * beta
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (ABz * CDy * beta * gamma * t8 - t4 * t5 * t6 - t4 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (CDz * delta * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 71:
        t0 = p + q
        t1 = CDx * gamma
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = CDz * delta
        t5 = ABz * beta
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (ABz * CDx * beta * gamma * t8 - t4 * t5 * t6 - t4 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (CDz * delta * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 72:
        t0 = p + q
        t1 = ABz * beta
        t2 = CDy * delta
        t3 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t4 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (-t0 ** 2.0 * t3 * t4 * F2 + t0 ** 3.0 * (t1 * t3 - t2 * t4) * F1 + t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 73:
        t0 = p + q
        t1 = CDy * delta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABz * beta
        t5 = CDz * gamma
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 - t1 * t2 * t3 + 2 * t2 * (ABz * CDz * beta * gamma * t6 - t4 * t9 - t5 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (CDy * delta * t8 - t4 * t6 - t5 * t6)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 74:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABz * beta
        t3 = q * t2
        t4 = t0 ** 13.5 * F0
        t5 = t0 ** 12.5 * F1
        t6 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t7 = q * t6
        t8 = t0 ** 11.5 * F2
        t9 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t10 = t2 * t9
        t11 = CDy * t6
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * CDy ** 2 * t1 * t2 * t4 + 2 * CDy * t5 * (ABz * beta * gamma * t9 - delta * gamma * t11 - delta * t10) + p * t3 * t5 - p * t7 * t8 + 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - t3 * t4 + t5 * t7 + 2 * t8 * t9 * (CDy * delta * t6 - gamma * t11 - t10)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 75:
        t0 = p + q
        t1 = CDx * gamma
        t2 = CDy * delta
        t3 = ABz * beta
        t4 = t2 * t3
        t5 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t8 = t1 * t7
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (CDy * delta * t6 * t7 - t3 * t5 * t6 - t5 * t8) * F2 + t0 ** 6.5 * (ABz * CDx * beta * gamma * t5 - t2 * t8 - t4 * t6) * F1 + t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 76:
        t0 = p + q
        t1 = ABz * beta
        t2 = CDx * delta
        t3 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t4 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (-t0 ** 2.0 * t3 * t4 * F2 + t0 ** 3.0 * (t1 * t3 - t2 * t4) * F1 + t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 77:
        t0 = p + q
        t1 = CDx * delta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABz * beta
        t5 = CDz * gamma
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 - t1 * t2 * t3 + 2 * t2 * (ABz * CDz * beta * gamma * t6 - t4 * t9 - t5 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (CDx * delta * t8 - t4 * t6 - t5 * t6)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 78:
        t0 = p + q
        t1 = CDy * gamma
        t2 = CDx * delta
        t3 = ABz * beta
        t4 = t2 * t3
        t5 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t8 = t1 * t7
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (CDx * delta * t6 * t7 - t3 * t5 * t6 - t5 * t8) * F2 + t0 ** 6.5 * (ABz * CDy * beta * gamma * t5 - t2 * t8 - t4 * t6) * F1 + t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 79:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABz * beta
        t3 = q * t2
        t4 = t0 ** 13.5 * F0
        t5 = t0 ** 12.5 * F1
        t6 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t7 = q * t6
        t8 = t0 ** 11.5 * F2
        t9 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t10 = t2 * t9
        t11 = CDx * t6
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * CDx ** 2 * t1 * t2 * t4 + 2 * CDx * t5 * (ABz * beta * gamma * t9 - delta * gamma * t11 - delta * t10) + p * t3 * t5 - p * t7 * t8 + 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - t3 * t4 + t5 * t7 + 2 * t8 * t9 * (CDx * delta * t6 - gamma * t11 - t10)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 80:
        t0 = q ** (-1.0)
        t1 = p + q
        t2 = alpha * beta
        t3 = F0
        t4 = t1 ** 6.0
        t5 = t1 ** 5.0 * F1
        t6 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        return np.pi ** 2.5 * t0 * t1 ** (-6.5) * (-2 * ABz ** 2 * t2 * t3 * t4 - 2 * ABz * t5 * t6 * (alpha - beta) - p * q * t5 + p * t3 * t4 + 2 * t1 ** 4.0 * t6 ** 2 * F2) * KAB * KCD / p ** 3
    if case_id == 81:
        t0 = p + q
        t1 = F0
        t2 = t0 ** 13.5
        t3 = CDz * gamma
        t4 = alpha * t3
        t5 = -ABz * alpha + ABz * beta + t3
        t6 = t0 ** 12.5
        t7 = F1
        t8 = t6 * t7
        t9 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t10 = t0 ** 11.5 * F2
        return np.pi ** 2.5 * t0 ** (-14.0) * (-2 * ABz ** 2 * beta * t1 * t2 * t4 - 2 * ABz * t8 * t9 * (ABz * alpha * beta - beta * t3 + t4) + CDz * gamma * p * t1 * t2 - 3 * p * q * t10 * t9 - p * q * t5 * t8 + p * t6 * t7 * t9 + 2 * t0 ** 10.5 * t9 ** 3 * F3 + 2 * t10 * t5 * t9 ** 2) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 82:
        t0 = p + q
        t1 = alpha * beta
        t2 = F0
        t3 = t0 ** 13.5
        t4 = t0 ** 12.5
        t5 = F1
        t6 = t4 * t5
        t7 = CDy * gamma
        t8 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t9 = t0 ** 11.5 * F2
        t10 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t11 = ABz * t8
        t12 = t10 * t7
        return np.pi ** 2.5 * t0 ** (-14.0) * (-2 * ABz ** 2 * t1 * t2 * t3 * t7 - 2 * ABz * t6 * (alpha * beta * t11 + alpha * t12 - beta * t12) + CDy * gamma * p * t2 * t3 - p * q * t6 * t7 - p * q * t8 * t9 + p * t4 * t5 * t8 + 2 * t0 ** 10.5 * t10 ** 2 * t8 * F3 - 2 * t10 * t9 * (ABz * alpha * t8 - beta * t11 - t12)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 83:
        t0 = p + q
        t1 = alpha * beta
        t2 = F0
        t3 = t0 ** 13.5
        t4 = t0 ** 12.5
        t5 = F1
        t6 = t4 * t5
        t7 = CDx * gamma
        t8 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t9 = t0 ** 11.5 * F2
        t10 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t11 = ABz * t8
        t12 = t10 * t7
        return np.pi ** 2.5 * t0 ** (-14.0) * (-2 * ABz ** 2 * t1 * t2 * t3 * t7 - 2 * ABz * t6 * (alpha * beta * t11 + alpha * t12 - beta * t12) + CDx * gamma * p * t2 * t3 - p * q * t6 * t7 - p * q * t8 * t9 + p * t4 * t5 * t8 + 2 * t0 ** 10.5 * t10 ** 2 * t8 * F3 - 2 * t10 * t9 * (ABz * alpha * t8 - beta * t11 - t12)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 84:
        t0 = p + q
        t1 = CDz * delta
        t2 = t0 ** 13.5 * F0
        t3 = beta * t1
        t4 = ABz * alpha - ABz * beta + t1
        t5 = t0 ** 12.5 * F1
        t6 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t7 = p * t6
        t8 = t0 ** 11.5 * F2
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * ABz ** 2 * alpha * t2 * t3 + 2 * ABz * t5 * t6 * (ABz * alpha * beta - alpha * t1 + t3) + p * q * t4 * t5 - p * t1 * t2 + 3 * q * t7 * t8 - 2 * t0 ** 10.5 * t6 ** 3 * F3 - 2 * t4 * t6 ** 2 * t8 - t5 * t7) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 85:
        t0 = p + q
        t1 = alpha * beta
        t2 = delta * gamma
        t3 = 1 / 2 * p
        t4 = t0 ** 34.5 * F0
        t5 = q * t4
        t6 = q ** 2
        t7 = t0 ** 33.5 * F1
        t8 = p ** 2
        t9 = q * t7
        t10 = t0 ** 32.5 * F2
        t11 = ABz ** 2 * t1
        t12 = CDz ** 2 * t2
        t13 = t12 * t4
        t14 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t15 = t14 ** 2
        t16 = p * t15
        t17 = t10 * t15
        t18 = t0 ** 31.5 * F3
        t19 = ABz * t14
        t20 = CDz * t7
        t21 = p * t14
        t22 = ABz * alpha
        t23 = CDz * gamma
        t24 = CDz * delta
        t25 = ABz * beta
        t26 = -ABz * CDz * alpha * delta - ABz * CDz * beta * gamma + t11 + t12 + t22 * t23 + t24 * t25
        t27 = t22 - t23 + t24 - t25
        return np.pi ** 2.5 * t0 ** (-35.0) * (-p * t13 + p * t26 * t9 - 3 * q * t10 * t21 * t27 - 6 * q * t16 * t18 + q * t17 + 2 * t0 ** 30.5 * t14 ** 4 * F4 + t10 * t16 + 3 / 2 * t10 * t6 * t8 + 2 * t11 * t13 - t11 * t5 + 2 * t14 ** 3 * t18 * t27 - 2 * t17 * t26 + 2 * t19 * t20 * (ABz * alpha * beta * gamma + CDz * beta * delta * gamma - alpha * gamma * t24 - beta * delta * t22) + t19 * t9 * (alpha - beta) + t20 * t21 * (delta - gamma) + t3 * t5 - t3 * t6 * t7 - 1 / 2 * t8 * t9) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 86:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDz * delta
        t3 = CDy * gamma
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = ABz ** 2 * t1
        t6 = ABz * alpha
        t7 = ABz * beta
        t8 = t0 ** 23.0 * F1
        t9 = p * t8
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t11 * t3
        t17 = t10 * t2
        t18 = t10 * t6 - t10 * t7 + t17
        t19 = t16 * t6
        t20 = t16 * t2
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * ABz * t8 * (-alpha * t20 - beta * t17 * t6 + beta * t19 + beta * t20) + p * t10 * t13 - p * t4 + q * t3 * t9 * (t2 + t6 - t7) + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (-t16 + t18) + t12 * t15 * (3 * CDy * gamma * t11 - t18) + 2 * t13 * (ABz * CDy * beta * gamma * t11 + ABz * CDz * alpha * delta * t10 - t10 * t5 - t17 * t7 - t19 - t20) + 2 * t4 * t5 + t9 * (-t16 + t17)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 87:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDz * delta
        t3 = CDx * gamma
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = ABz ** 2 * t1
        t6 = ABz * alpha
        t7 = ABz * beta
        t8 = t0 ** 23.0 * F1
        t9 = p * t8
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t11 * t3
        t17 = t10 * t2
        t18 = t10 * t6 - t10 * t7 + t17
        t19 = t16 * t6
        t20 = t16 * t2
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * ABz * t8 * (-alpha * t20 - beta * t17 * t6 + beta * t19 + beta * t20) + p * t10 * t13 - p * t4 + q * t3 * t9 * (t2 + t6 - t7) + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (-t16 + t18) + t12 * t15 * (3 * CDx * gamma * t11 - t18) + 2 * t13 * (ABz * CDx * beta * gamma * t11 + ABz * CDz * alpha * delta * t10 - t10 * t5 - t17 * t7 - t19 - t20) + 2 * t4 * t5 + t9 * (-t16 + t17)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 88:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDy * delta
        t3 = p * t2
        t4 = t0 ** 13.5 * F0
        t5 = t0 ** 12.5 * F1
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = p * t6
        t8 = t0 ** 11.5 * F2
        t9 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t10 = ABz * t6
        t11 = t2 * t9
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * ABz ** 2 * t1 * t2 * t4 + 2 * ABz * t5 * (alpha * beta * t10 - alpha * t11 + beta * t11) + q * t3 * t5 + q * t7 * t8 - 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - t3 * t4 - t5 * t7 + 2 * t8 * t9 * (ABz * beta * t6 - alpha * t10 - t11)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 89:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDz * gamma
        t3 = CDy * delta
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = ABz ** 2 * t1
        t6 = ABz * beta
        t7 = ABz * alpha
        t8 = t0 ** 23.0 * F1
        t9 = p * t8
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t11 * t3
        t17 = t10 * t2
        t18 = -t17
        t19 = t16 + t18
        t20 = -t10 * t6 + t10 * t7
        t21 = t16 * t6
        t22 = t16 * t2
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * ABz * t8 * (ABz * CDz * alpha * beta * gamma * t10 + CDy * CDz * beta * delta * gamma * t11 - alpha * t21 - alpha * t22) + p * t10 * t13 - p * t4 + q * t3 * t9 * (t2 + t6 - t7) + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (t19 + t20) + t12 * t15 * (-3 * t16 - t18 - t20) + 2 * t13 * (ABz * CDy * alpha * delta * t11 + ABz * CDz * beta * gamma * t10 - t10 * t5 - t17 * t7 - t21 - t22) + t19 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 90:
        t0 = p + q
        t1 = alpha * beta
        t2 = delta * gamma
        t3 = 1 / 2 * p
        t4 = t0 ** 34.5 * F0
        t5 = q * t4
        t6 = q ** 2
        t7 = t0 ** 33.5 * F1
        t8 = 1 / 2 * p ** 2
        t9 = q * t7
        t10 = t0 ** 32.5 * F2
        t11 = ABz ** 2 * t1
        t12 = CDy ** 2 * t2
        t13 = t12 * t4
        t14 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t15 = t14 ** 2
        t16 = p * t10
        t17 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t18 = t17 ** 2
        t19 = ABz * t17
        t20 = CDy * t14
        t21 = t0 ** 31.5 * F3
        t22 = alpha * t19
        t23 = delta * t20
        t24 = ABz * t14
        t25 = CDy * t17
        return np.pi ** 2.5 * t0 ** (-35.0) * (2 * ABz * CDy * t7 * (ABz * alpha * beta * gamma * t14 + CDy * beta * delta * gamma * t17 - alpha * t2 * t25 - delta * t1 * t24) - p * q * t21 * (t15 + t18) - p * t13 + p * t20 * t7 * (delta - gamma) + p * t9 * (t11 + t12) + q * t10 * t18 + q * t16 * (ABz * beta * t17 + CDy * gamma * t14 - t22 - t23) + 2 * t0 ** 30.5 * t15 * t18 * F4 + t10 * t6 * t8 + 2 * t10 * (ABz * CDy * alpha * delta * t14 * t17 + ABz * CDy * beta * gamma * t14 * t17 - beta * t19 * t23 - gamma * t20 * t22 - t11 * t15 - t12 * t18) + 2 * t11 * t13 - t11 * t5 + 2 * t14 * t17 * t21 * (alpha * t24 - beta * t24 + delta * t25 - gamma * t25) + t15 * t16 + t19 * t9 * (alpha - beta) + t3 * t5 - t3 * t6 * t7 - t8 * t9) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 91:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDy * delta
        t3 = CDx * gamma
        t4 = t2 * t3
        t5 = t0 ** 24.0 * t4 * F0
        t6 = t0 ** 23.0 * F1
        t7 = p * t6
        t8 = ABz ** 2
        t9 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = t10 * t9
        t12 = t0 ** 22.0 * F2
        t13 = p * t12
        t14 = t0 ** 21.0 * F3
        t15 = t10 * t3
        t16 = -CDy * delta * t9 + t15
        t17 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t18 = t17 ** 2
        t19 = ABz * alpha
        t20 = beta * t19
        t21 = t17 * t4
        t22 = t2 * t9
        t23 = t17 * t22
        t24 = beta * t11
        t25 = t15 * t17
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * ABz * t6 * (-alpha * t21 + beta * t21 + t15 * t20 - t20 * t22) - p * q * t11 * t14 - p * t5 + q * t13 * t16 + q * t4 * t7 + 2 * t0 ** 20.0 * t11 * t18 * F4 + 2 * t1 * t5 * t8 + t11 * t13 + 2 * t12 * (ABz * CDx * beta * gamma * t10 * t17 + ABz * CDy * alpha * delta * t17 * t9 - ABz * beta * t23 - alpha * t24 * t8 - t18 * t4 - t19 * t25) + 2 * t14 * t17 * (-ABz * t24 + t11 * t19 + t23 - t25) - t16 * t7) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 92:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDx * delta
        t3 = p * t2
        t4 = t0 ** 13.5 * F0
        t5 = t0 ** 12.5 * F1
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = p * t6
        t8 = t0 ** 11.5 * F2
        t9 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t10 = ABz * t6
        t11 = t2 * t9
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * ABz ** 2 * t1 * t2 * t4 + 2 * ABz * t5 * (alpha * beta * t10 - alpha * t11 + beta * t11) + q * t3 * t5 + q * t7 * t8 - 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - t3 * t4 - t5 * t7 + 2 * t8 * t9 * (ABz * beta * t6 - alpha * t10 - t11)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 93:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDz * gamma
        t3 = CDx * delta
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = ABz ** 2 * t1
        t6 = ABz * beta
        t7 = ABz * alpha
        t8 = t0 ** 23.0 * F1
        t9 = p * t8
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t11 * t3
        t17 = t10 * t2
        t18 = -t17
        t19 = t16 + t18
        t20 = -t10 * t6 + t10 * t7
        t21 = t16 * t6
        t22 = t16 * t2
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * ABz * t8 * (ABz * CDz * alpha * beta * gamma * t10 + CDx * CDz * beta * delta * gamma * t11 - alpha * t21 - alpha * t22) + p * t10 * t13 - p * t4 + q * t3 * t9 * (t2 + t6 - t7) + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (t19 + t20) + t12 * t15 * (-3 * t16 - t18 - t20) + 2 * t13 * (ABz * CDx * alpha * delta * t11 + ABz * CDz * beta * gamma * t10 - t10 * t5 - t17 * t7 - t21 - t22) + t19 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 94:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDx * delta
        t3 = CDy * gamma
        t4 = t2 * t3
        t5 = t0 ** 24.0 * t4 * F0
        t6 = t0 ** 23.0 * F1
        t7 = p * t6
        t8 = ABz ** 2
        t9 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = t10 * t9
        t12 = t0 ** 22.0 * F2
        t13 = p * t12
        t14 = t0 ** 21.0 * F3
        t15 = t2 * t9
        t16 = t10 * t3
        t17 = t15 - t16
        t18 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t19 = t18 ** 2
        t20 = ABz * alpha
        t21 = t15 * t18
        t22 = beta * t11
        t23 = t16 * t18
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * ABz * t6 * (ABz * CDy * alpha * beta * gamma * t10 + CDx * CDy * beta * delta * gamma * t18 - alpha * t18 * t4 - beta * t15 * t20) - p * q * t11 * t14 - p * t5 - q * t13 * t17 + q * t4 * t7 + 2 * t0 ** 20.0 * t11 * t19 * F4 + 2 * t1 * t5 * t8 + t11 * t13 + 2 * t12 * (ABz * CDx * alpha * delta * t18 * t9 + ABz * CDy * beta * gamma * t10 * t18 - ABz * beta * t21 - alpha * t22 * t8 - t19 * t4 - t20 * t23) + 2 * t14 * t18 * (-ABz * t22 + t11 * t20 + t21 - t23) + t17 * t7) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 95:
        t0 = p + q
        t1 = alpha * beta
        t2 = delta * gamma
        t3 = 1 / 2 * p
        t4 = t0 ** 34.5 * F0
        t5 = q * t4
        t6 = q ** 2
        t7 = t0 ** 33.5 * F1
        t8 = 1 / 2 * p ** 2
        t9 = q * t7
        t10 = t0 ** 32.5 * F2
        t11 = ABz ** 2 * t1
        t12 = CDx ** 2 * t2
        t13 = t12 * t4
        t14 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t15 = t14 ** 2
        t16 = p * t10
        t17 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t18 = t17 ** 2
        t19 = ABz * t17
        t20 = CDx * t14
        t21 = t0 ** 31.5 * F3
        t22 = alpha * t19
        t23 = delta * t20
        t24 = ABz * t14
        t25 = CDx * t17
        return np.pi ** 2.5 * t0 ** (-35.0) * (2 * ABz * CDx * t7 * (ABz * alpha * beta * gamma * t14 + CDx * beta * delta * gamma * t17 - alpha * t2 * t25 - delta * t1 * t24) - p * q * t21 * (t15 + t18) - p * t13 + p * t20 * t7 * (delta - gamma) + p * t9 * (t11 + t12) + q * t10 * t18 + q * t16 * (ABz * beta * t17 + CDx * gamma * t14 - t22 - t23) + 2 * t0 ** 30.5 * t15 * t18 * F4 + t10 * t6 * t8 + 2 * t10 * (ABz * CDx * alpha * delta * t14 * t17 + ABz * CDx * beta * gamma * t14 * t17 - beta * t19 * t23 - gamma * t20 * t22 - t11 * t15 - t12 * t18) + 2 * t11 * t13 - t11 * t5 + 2 * t14 * t17 * t21 * (alpha * t24 - beta * t24 + delta * t25 - gamma * t25) + t15 * t16 + t19 * t9 * (alpha - beta) + t3 * t5 - t3 * t6 * t7 - t8 * t9) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 96:
        t0 = q ** (-1.0)
        t1 = p + q
        t2 = ABy * alpha
        t3 = ABz * beta
        t4 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t5 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-4.5) * (t1 ** 2.0 * t4 * t5 * F2 - t1 ** 3.0 * (t2 * t5 - t3 * t4) * F1 - t1 ** 4.0 * t2 * t3 * F0) * KAB * KCD / p ** 3
    if case_id == 97:
        t0 = p + q
        t1 = F1
        t2 = t0 ** 10.5
        t3 = ABy * alpha
        t4 = ABz * beta
        t5 = CDz * gamma
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = t0 ** 9.5 * F2
        t8 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t9 = t3 * t8
        t10 = t4 * t6
        return np.pi ** 2.5 * t0 ** (-12.0) * (ABy * alpha * p * q * t1 * t2 - p * q * t6 * t7 + 2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t3 * t4 * t5 * F0 - 2 * t1 * t2 * (-t10 * t5 + t4 * t9 + t5 * t9) - 2 * t7 * t8 * (ABy * alpha * t8 - t10 - t5 * t6)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 98:
        t0 = p + q
        t1 = ABz * beta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABy * alpha
        t5 = CDy * gamma
        t6 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t7 = t0 ** 9.5 * F2
        t8 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 - t1 * t2 * t3 - 2 * t2 * (t4 * t5 * t6 + t4 * t9 - t5 * t9) - t3 * t6 * t7 - 2 * t7 * t8 * (ABy * alpha * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 99:
        t0 = p + q
        t1 = ABy * alpha
        t2 = ABz * beta
        t3 = CDx * gamma
        t4 = t2 * t3
        t5 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t8 = t2 * t5
        t9 = t3 * t7
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 - t0 ** 5.5 * (ABy * alpha * t5 * t7 - t6 * t8 - t6 * t9) * F2 - t0 ** 6.5 * (t1 * t8 + t1 * t9 - t4 * t6) * F1 - t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 100:
        t0 = p + q
        t1 = ABy * alpha
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABz * beta
        t5 = CDz * delta
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (-2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (t4 * t5 * t6 + t4 * t9 - t5 * t9) + t3 * t6 * t7 + 2 * t7 * t8 * (ABz * beta * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 101:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABz * beta
        t3 = ABy * alpha
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = CDz ** 2 * t1
        t6 = CDz * gamma
        t7 = CDz * delta
        t8 = t0 ** 23.0 * F1
        t9 = q * t8
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t11 * t3
        t17 = t10 * t2
        t18 = -t17
        t19 = t16 + t18
        t20 = -t10 * t6 + t10 * t7
        t21 = t16 * t2
        t22 = t16 * t6
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDz * t8 * (ABy * ABz * alpha * beta * gamma * t11 + ABz * CDz * beta * delta * gamma * t10 - delta * t21 - delta * t22) + p * t3 * t9 * (t2 + t6 - t7) + q * t10 * t13 - q * t4 + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (t19 + t20) + t12 * t15 * (-3 * t16 - t18 - t20) + 2 * t13 * (ABy * CDz * alpha * delta * t11 + ABz * CDz * beta * gamma * t10 - t10 * t5 - t17 * t7 - t21 - t22) + t19 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 102:
        t0 = p + q
        t1 = t0 ** 20.0 * F2
        t2 = CDy * gamma
        t3 = ABy * alpha
        t4 = t2 * t3
        t5 = ABz * beta
        t6 = CDz * delta
        t7 = t5 * t6
        t8 = t0 ** 21.0 * F1
        t9 = p * q
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = t10 ** 2
        t12 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t13 = t12 ** 2
        t14 = t0 ** 19.0 * F3
        t15 = t10 * t3
        t16 = t12 * t6
        t17 = t12 * t5
        t18 = t10 * t2
        return np.pi ** 2.5 * t0 ** (-22.5) * (1 / 2 * p ** 2 * q ** 2 * t1 + 2 * t0 ** 18.0 * t11 * t13 * F4 + 2 * t0 ** 22.0 * t4 * t7 * F0 + t1 * t9 * (ABz * beta * t12 + CDy * gamma * t10 - t15 - t16) + 2 * t1 * (ABy * CDz * alpha * delta * t10 * t12 + ABz * CDy * beta * gamma * t10 * t12 - t11 * t7 - t13 * t4 - t15 * t17 - t16 * t18) + 2 * t10 * t12 * t14 * (-t10 * t5 + t10 * t6 - t12 * t2 + t12 * t3) - t14 * t9 * (t11 + t13) + t8 * t9 * (t4 + t7) + 2 * t8 * (-t15 * t7 - t16 * t4 + t17 * t4 + t18 * t7)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 103:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = CDx * gamma
        t3 = ABy * alpha
        t4 = t2 * t3
        t5 = p * q
        t6 = CDz * delta
        t7 = ABz * beta
        t8 = t6 * t7
        t9 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = t10 * t9
        t12 = t0 ** 16.5 * F3
        t13 = t10 * t3
        t14 = t0 ** 17.5 * F2
        t15 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t16 = t15 ** 2
        t17 = t15 * t4
        t18 = t2 * t9
        t19 = t13 * t15
        t20 = t11 * t6
        t21 = t15 * t18
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t11 * t16 * F4 + 2 * t0 ** 19.5 * t4 * t8 * F0 + t1 * t4 * t5 + 2 * t1 * (-t13 * t8 - t17 * t6 + t17 * t7 + t18 * t8) - t11 * t12 * t5 + 2 * t12 * t15 * (-t11 * t7 + t19 + t20 - t21) + t14 * t5 * (CDx * gamma * t9 - t13) + 2 * t14 * (ABy * CDz * alpha * delta * t10 * t15 + ABz * CDx * beta * gamma * t15 * t9 - t16 * t4 - t19 * t7 - t20 * t7 - t21 * t6)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 104:
        t0 = p + q
        t1 = ABz * beta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t5 = t0 ** 9.5 * F2
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = ABy * alpha
        t8 = t1 * t6
        t9 = CDy * delta
        t10 = t4 * t7
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * ABy * ABz * CDy * alpha * beta * delta * t0 ** 11.5 * F0 + 2 * t0 ** 8.5 * t4 * t6 ** 2 * F3 - t1 * t2 * t3 - 2 * t2 * (-t10 * t9 + t7 * t8 + t8 * t9) - t3 * t4 * t5 - 2 * t5 * t6 * (t10 + t4 * t9 - t8)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 105:
        t0 = p + q
        t1 = F2
        t2 = t0 ** 20.0
        t3 = ABy * alpha
        t4 = CDy * delta
        t5 = t3 * t4
        t6 = ABz * beta
        t7 = CDz * gamma
        t8 = t6 * t7
        t9 = t0 ** 21.0 * F1
        t10 = p * q
        t11 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t12 = t11 ** 2
        t13 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t14 = t13 ** 2
        t15 = t0 ** 19.0 * F3
        t16 = t11 * t3
        t17 = t11 * t4
        t18 = t1 * t2
        t19 = t13 * t6
        t20 = t13 * t7
        return np.pi ** 2.5 * t0 ** (-22.5) * (2 * ABy * ABz * CDy * CDz * alpha * beta * delta * gamma * t0 ** 22.0 * F0 + 1 / 2 * p ** 2 * q ** 2 * t1 * t2 + 2 * t0 ** 18.0 * t12 * t14 * F4 - t10 * t15 * (t12 + t14) - t10 * t18 * (ABz * beta * t13 + CDz * gamma * t13 - t16 - t17) - t10 * t9 * (t5 + t8) - 2 * t11 * t13 * t15 * (-t11 * t6 - t11 * t7 + t13 * t3 + t13 * t4) - 2 * t18 * (-t12 * t8 - t14 * t5 + t16 * t19 + t16 * t20 + t17 * t19 + t17 * t20) - 2 * t9 * (ABy * ABz * CDz * alpha * beta * gamma * t11 + ABz * CDy * CDz * beta * delta * gamma * t11 - t19 * t5 - t20 * t5)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 106:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABy * alpha
        t3 = ABz * beta
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = CDy ** 2 * t1
        t6 = CDy * delta
        t7 = CDy * gamma
        t8 = t0 ** 23.0 * F1
        t9 = q * t8
        t10 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t11 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t10 * t2
        t17 = t11 * t3
        t18 = t16 - t17
        t19 = t10 * t6 - t10 * t7
        t20 = t17 * t2
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDy * t8 * (ABy * ABz * alpha * beta * gamma * t11 + ABz * CDy * beta * delta * gamma * t11 - delta * t20 - gamma * t16 * t6) + p * t3 * t9 * (t2 + t6 - t7) + q * t10 * t13 - q * t4 + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (t18 + t19) + t12 * t15 * (3 * ABz * beta * t11 - t16 - t19) + 2 * t13 * (ABy * CDy * alpha * delta * t10 + ABz * CDy * beta * gamma * t11 - t10 * t5 - t16 * t7 - t17 * t6 - t20) + t18 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 107:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABz * beta
        t3 = CDx * gamma
        t4 = t2 * t3
        t5 = p * q
        t6 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t7 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t8 = t6 * t7
        t9 = t0 ** 16.5 * F3
        t10 = t2 * t6
        t11 = t3 * t7
        t12 = t0 ** 17.5 * F2
        t13 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t14 = t13 ** 2
        t15 = ABy * alpha
        t16 = t13 * t4
        t17 = CDy * delta
        t18 = t15 * t17
        t19 = t15 * t8
        t20 = t10 * t13
        t21 = t11 * t13
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * ABy * ABz * CDx * CDy * alpha * beta * delta * gamma * t0 ** 19.5 * F0 + 2 * t0 ** 15.5 * t14 * t6 * t7 * F4 - t1 * t4 * t5 - 2 * t1 * (-t10 * t18 - t11 * t18 + t15 * t16 + t16 * t17) - t12 * t5 * (t10 + t11) - 2 * t12 * (-t14 * t4 + t15 * t20 + t15 * t21 - t17 * t19 + t17 * t20 + t17 * t21) - 2 * t13 * t9 * (t17 * t8 + t19 - t20 - t21) - t5 * t8 * t9) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 108:
        t0 = p + q
        t1 = ABz * beta
        t2 = ABy * alpha
        t3 = CDx * delta
        t4 = t2 * t3
        t5 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t8 = t2 * t6
        t9 = t3 * t7
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (-t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (ABz * beta * t6 * t7 - t5 * t8 - t5 * t9) * F2 + t0 ** 6.5 * (t1 * t8 + t1 * t9 - t4 * t5) * F1 + t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 109:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABy * alpha
        t3 = CDx * delta
        t4 = t2 * t3
        t5 = p * q
        t6 = ABz * beta
        t7 = CDz * gamma
        t8 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t9 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t10 = t8 * t9
        t11 = t0 ** 16.5 * F3
        t12 = t2 * t8
        t13 = t3 * t9
        t14 = t0 ** 17.5 * F2
        t15 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t16 = t15 ** 2
        t17 = t15 * t4
        t18 = t12 * t15
        t19 = t13 * t15
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t10 * t16 * F4 + 2 * t0 ** 19.5 * t4 * t6 * t7 * F0 - t1 * t4 * t5 + 2 * t1 * (ABy * ABz * CDz * alpha * beta * gamma * t8 + ABz * CDx * CDz * beta * delta * gamma * t9 - t17 * t6 - t17 * t7) - t10 * t11 * t5 + 2 * t11 * t15 * (-t10 * t6 - t10 * t7 + t18 + t19) - t14 * t5 * (t12 + t13) + 2 * t14 * (ABy * CDx * alpha * delta * t16 + ABz * CDz * beta * gamma * t8 * t9 - t18 * t6 - t18 * t7 - t19 * t6 - t19 * t7)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 110:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABz * beta
        t3 = CDx * delta
        t4 = t2 * t3
        t5 = p * q
        t6 = ABy * alpha
        t7 = CDy * gamma
        t8 = t6 * t7
        t9 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t10 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t11 = t10 * t9
        t12 = t0 ** 16.5 * F3
        t13 = t2 * t9
        t14 = t10 * t3
        t15 = t0 ** 17.5 * F2
        t16 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t17 = t16 ** 2
        t18 = t11 * t6
        t19 = t14 * t16
        t20 = t13 * t16
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t11 * t17 * F4 + 2 * t0 ** 19.5 * t4 * t8 * F0 + t1 * t4 * t5 + 2 * t1 * (ABy * ABz * CDy * alpha * beta * gamma * t9 + ABz * CDx * CDy * beta * delta * gamma * t16 - t14 * t8 - t16 * t4 * t6) - t11 * t12 * t5 + 2 * t12 * t16 * (-t11 * t7 + t18 + t19 - t20) + t15 * t5 * (t13 - t14) + 2 * t15 * (ABy * CDx * alpha * delta * t10 * t16 + ABz * CDy * beta * gamma * t16 * t9 - t17 * t4 - t18 * t7 - t19 * t7 - t20 * t6)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 111:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABy * alpha
        t3 = ABz * beta
        t4 = t2 * t3
        t5 = t0 ** 24.0 * t4 * F0
        t6 = t0 ** 23.0 * F1
        t7 = q * t6
        t8 = CDx ** 2
        t9 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = t10 * t9
        t12 = t0 ** 22.0 * F2
        t13 = q * t12
        t14 = t0 ** 21.0 * F3
        t15 = t2 * t9
        t16 = t10 * t3
        t17 = t15 - t16
        t18 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t19 = t18 ** 2
        t20 = CDx * delta
        t21 = t15 * t18
        t22 = t16 * t18
        t23 = gamma * t11
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDx * t6 * (ABy * ABz * alpha * beta * gamma * t18 + ABz * CDx * beta * delta * gamma * t10 - delta * t18 * t4 - gamma * t15 * t20) - p * q * t11 * t14 - p * t13 * t17 + p * t4 * t7 - q * t5 + 2 * t0 ** 20.0 * t11 * t19 * F4 + 2 * t1 * t5 * t8 + t11 * t13 + 2 * t12 * (ABy * CDx * alpha * delta * t18 * t9 + ABz * CDx * beta * gamma * t10 * t18 - CDx * gamma * t21 - delta * t23 * t8 - t19 * t4 - t20 * t22) + 2 * t14 * t18 * (-CDx * t23 + t11 * t20 + t21 - t22) + t17 * t7) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 112:
        t0 = q ** (-1.0)
        t1 = p + q
        t2 = ABx * alpha
        t3 = ABz * beta
        t4 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t5 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-4.5) * (t1 ** 2.0 * t4 * t5 * F2 - t1 ** 3.0 * (t2 * t5 - t3 * t4) * F1 - t1 ** 4.0 * t2 * t3 * F0) * KAB * KCD / p ** 3
    if case_id == 113:
        t0 = p + q
        t1 = F1
        t2 = t0 ** 10.5
        t3 = ABx * alpha
        t4 = ABz * beta
        t5 = CDz * gamma
        t6 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t7 = t0 ** 9.5 * F2
        t8 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t9 = t3 * t8
        t10 = t4 * t6
        return np.pi ** 2.5 * t0 ** (-12.0) * (ABx * alpha * p * q * t1 * t2 - p * q * t6 * t7 + 2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t3 * t4 * t5 * F0 - 2 * t1 * t2 * (-t10 * t5 + t4 * t9 + t5 * t9) - 2 * t7 * t8 * (ABx * alpha * t8 - t10 - t5 * t6)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 114:
        t0 = p + q
        t1 = ABx * alpha
        t2 = ABz * beta
        t3 = CDy * gamma
        t4 = t2 * t3
        t5 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t8 = t2 * t6
        t9 = t3 * t7
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 - t0 ** 5.5 * (ABx * alpha * t6 * t7 - t5 * t8 - t5 * t9) * F2 - t0 ** 6.5 * (t1 * t8 + t1 * t9 - t4 * t5) * F1 - t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 115:
        t0 = p + q
        t1 = ABz * beta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABx * alpha
        t5 = CDx * gamma
        t6 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t7 = t0 ** 9.5 * F2
        t8 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 - t1 * t2 * t3 - 2 * t2 * (t4 * t5 * t6 + t4 * t9 - t5 * t9) - t3 * t6 * t7 - 2 * t7 * t8 * (ABx * alpha * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 116:
        t0 = p + q
        t1 = ABx * alpha
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABz * beta
        t5 = CDz * delta
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (-2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (t4 * t5 * t6 + t4 * t9 - t5 * t9) + t3 * t6 * t7 + 2 * t7 * t8 * (ABz * beta * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 117:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABz * beta
        t3 = ABx * alpha
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = CDz ** 2 * t1
        t6 = CDz * gamma
        t7 = CDz * delta
        t8 = t0 ** 23.0 * F1
        t9 = q * t8
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t11 * t3
        t17 = t10 * t2
        t18 = -t17
        t19 = t16 + t18
        t20 = -t10 * t6 + t10 * t7
        t21 = t16 * t2
        t22 = t16 * t6
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDz * t8 * (ABx * ABz * alpha * beta * gamma * t11 + ABz * CDz * beta * delta * gamma * t10 - delta * t21 - delta * t22) + p * t3 * t9 * (t2 + t6 - t7) + q * t10 * t13 - q * t4 + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (t19 + t20) + t12 * t15 * (-3 * t16 - t18 - t20) + 2 * t13 * (ABx * CDz * alpha * delta * t11 + ABz * CDz * beta * gamma * t10 - t10 * t5 - t17 * t7 - t21 - t22) + t19 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 118:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = CDy * gamma
        t3 = ABx * alpha
        t4 = t2 * t3
        t5 = p * q
        t6 = CDz * delta
        t7 = ABz * beta
        t8 = t6 * t7
        t9 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = t10 * t9
        t12 = t0 ** 16.5 * F3
        t13 = t10 * t3
        t14 = t0 ** 17.5 * F2
        t15 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t16 = t15 ** 2
        t17 = t15 * t4
        t18 = t2 * t9
        t19 = t13 * t15
        t20 = t11 * t6
        t21 = t15 * t18
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t11 * t16 * F4 + 2 * t0 ** 19.5 * t4 * t8 * F0 + t1 * t4 * t5 + 2 * t1 * (-t13 * t8 - t17 * t6 + t17 * t7 + t18 * t8) - t11 * t12 * t5 + 2 * t12 * t15 * (-t11 * t7 + t19 + t20 - t21) + t14 * t5 * (CDy * gamma * t9 - t13) + 2 * t14 * (ABx * CDz * alpha * delta * t10 * t15 + ABz * CDy * beta * gamma * t15 * t9 - t16 * t4 - t19 * t7 - t20 * t7 - t21 * t6)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 119:
        t0 = p + q
        t1 = t0 ** 20.0 * F2
        t2 = CDx * gamma
        t3 = ABx * alpha
        t4 = t2 * t3
        t5 = ABz * beta
        t6 = CDz * delta
        t7 = t5 * t6
        t8 = t0 ** 21.0 * F1
        t9 = p * q
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = t10 ** 2
        t12 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t13 = t12 ** 2
        t14 = t0 ** 19.0 * F3
        t15 = t10 * t3
        t16 = t12 * t6
        t17 = t12 * t5
        t18 = t10 * t2
        return np.pi ** 2.5 * t0 ** (-22.5) * (1 / 2 * p ** 2 * q ** 2 * t1 + 2 * t0 ** 18.0 * t11 * t13 * F4 + 2 * t0 ** 22.0 * t4 * t7 * F0 + t1 * t9 * (ABz * beta * t12 + CDx * gamma * t10 - t15 - t16) + 2 * t1 * (ABx * CDz * alpha * delta * t10 * t12 + ABz * CDx * beta * gamma * t10 * t12 - t11 * t7 - t13 * t4 - t15 * t17 - t16 * t18) + 2 * t10 * t12 * t14 * (-t10 * t5 + t10 * t6 - t12 * t2 + t12 * t3) - t14 * t9 * (t11 + t13) + t8 * t9 * (t4 + t7) + 2 * t8 * (-t15 * t7 - t16 * t4 + t17 * t4 + t18 * t7)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 120:
        t0 = p + q
        t1 = ABz * beta
        t2 = ABx * alpha
        t3 = CDy * delta
        t4 = t2 * t3
        t5 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t8 = t2 * t7
        t9 = t3 * t6
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (-t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (ABz * beta * t6 * t7 - t5 * t8 - t5 * t9) * F2 + t0 ** 6.5 * (t1 * t8 + t1 * t9 - t4 * t5) * F1 + t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 121:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABx * alpha
        t3 = CDy * delta
        t4 = t2 * t3
        t5 = p * q
        t6 = ABz * beta
        t7 = CDz * gamma
        t8 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t9 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t10 = t8 * t9
        t11 = t0 ** 16.5 * F3
        t12 = t2 * t8
        t13 = t3 * t9
        t14 = t0 ** 17.5 * F2
        t15 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t16 = t15 ** 2
        t17 = t15 * t4
        t18 = t12 * t15
        t19 = t13 * t15
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t10 * t16 * F4 + 2 * t0 ** 19.5 * t4 * t6 * t7 * F0 - t1 * t4 * t5 + 2 * t1 * (ABx * ABz * CDz * alpha * beta * gamma * t8 + ABz * CDy * CDz * beta * delta * gamma * t9 - t17 * t6 - t17 * t7) - t10 * t11 * t5 + 2 * t11 * t15 * (-t10 * t6 - t10 * t7 + t18 + t19) - t14 * t5 * (t12 + t13) + 2 * t14 * (ABx * CDy * alpha * delta * t16 + ABz * CDz * beta * gamma * t8 * t9 - t18 * t6 - t18 * t7 - t19 * t6 - t19 * t7)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 122:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABx * alpha
        t3 = ABz * beta
        t4 = t2 * t3
        t5 = t0 ** 24.0 * t4 * F0
        t6 = t0 ** 23.0 * F1
        t7 = q * t6
        t8 = CDy ** 2
        t9 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = t10 * t9
        t12 = t0 ** 22.0 * F2
        t13 = q * t12
        t14 = t0 ** 21.0 * F3
        t15 = t2 * t9
        t16 = t10 * t3
        t17 = t15 - t16
        t18 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t19 = t18 ** 2
        t20 = CDy * delta
        t21 = t15 * t18
        t22 = t16 * t18
        t23 = gamma * t11
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDy * t6 * (ABx * ABz * alpha * beta * gamma * t18 + ABz * CDy * beta * delta * gamma * t10 - delta * t18 * t4 - gamma * t15 * t20) - p * q * t11 * t14 - p * t13 * t17 + p * t4 * t7 - q * t5 + 2 * t0 ** 20.0 * t11 * t19 * F4 + 2 * t1 * t5 * t8 + t11 * t13 + 2 * t12 * (ABx * CDy * alpha * delta * t18 * t9 + ABz * CDy * beta * gamma * t10 * t18 - CDy * gamma * t21 - delta * t23 * t8 - t19 * t4 - t20 * t22) + 2 * t14 * t18 * (-CDy * t23 + t11 * t20 + t21 - t22) + t17 * t7) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 123:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABz * beta
        t3 = CDy * delta
        t4 = t2 * t3
        t5 = p * q
        t6 = ABx * alpha
        t7 = CDx * gamma
        t8 = t6 * t7
        t9 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t10 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t11 = t10 * t9
        t12 = t0 ** 16.5 * F3
        t13 = t2 * t9
        t14 = t10 * t3
        t15 = t0 ** 17.5 * F2
        t16 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t17 = t16 ** 2
        t18 = t16 * t4
        t19 = t11 * t6
        t20 = t14 * t16
        t21 = t13 * t16
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t11 * t17 * F4 + 2 * t0 ** 19.5 * t4 * t8 * F0 + t1 * t4 * t5 + 2 * t1 * (t13 * t8 - t14 * t8 - t18 * t6 + t18 * t7) - t11 * t12 * t5 + 2 * t12 * t16 * (-t11 * t7 + t19 + t20 - t21) + t15 * t5 * (t13 - t14) + 2 * t15 * (ABx * CDy * alpha * delta * t10 * t16 + ABz * CDx * beta * gamma * t16 * t9 - t17 * t4 - t19 * t7 - t20 * t7 - t21 * t6)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 124:
        t0 = p + q
        t1 = ABz * beta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t5 = t0 ** 9.5 * F2
        t6 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t7 = ABx * alpha
        t8 = t1 * t6
        t9 = CDx * delta
        t10 = t4 * t7
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * ABx * ABz * CDx * alpha * beta * delta * t0 ** 11.5 * F0 + 2 * t0 ** 8.5 * t4 * t6 ** 2 * F3 - t1 * t2 * t3 - 2 * t2 * (-t10 * t9 + t7 * t8 + t8 * t9) - t3 * t4 * t5 - 2 * t5 * t6 * (t10 + t4 * t9 - t8)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 125:
        t0 = p + q
        t1 = F2
        t2 = t0 ** 20.0
        t3 = ABx * alpha
        t4 = CDx * delta
        t5 = t3 * t4
        t6 = ABz * beta
        t7 = CDz * gamma
        t8 = t6 * t7
        t9 = t0 ** 21.0 * F1
        t10 = p * q
        t11 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t12 = t11 ** 2
        t13 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t14 = t13 ** 2
        t15 = t0 ** 19.0 * F3
        t16 = t11 * t3
        t17 = t11 * t4
        t18 = t1 * t2
        t19 = t13 * t6
        t20 = t13 * t7
        return np.pi ** 2.5 * t0 ** (-22.5) * (2 * ABx * ABz * CDx * CDz * alpha * beta * delta * gamma * t0 ** 22.0 * F0 + 1 / 2 * p ** 2 * q ** 2 * t1 * t2 + 2 * t0 ** 18.0 * t12 * t14 * F4 - t10 * t15 * (t12 + t14) - t10 * t18 * (ABz * beta * t13 + CDz * gamma * t13 - t16 - t17) - t10 * t9 * (t5 + t8) - 2 * t11 * t13 * t15 * (-t11 * t6 - t11 * t7 + t13 * t3 + t13 * t4) - 2 * t18 * (-t12 * t8 - t14 * t5 + t16 * t19 + t16 * t20 + t17 * t19 + t17 * t20) - 2 * t9 * (ABx * ABz * CDz * alpha * beta * gamma * t11 + ABz * CDx * CDz * beta * delta * gamma * t11 - t19 * t5 - t20 * t5)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 126:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABz * beta
        t3 = CDy * gamma
        t4 = t2 * t3
        t5 = p * q
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t8 = t6 * t7
        t9 = t0 ** 16.5 * F3
        t10 = t2 * t6
        t11 = t3 * t7
        t12 = t0 ** 17.5 * F2
        t13 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t14 = t13 ** 2
        t15 = ABx * alpha
        t16 = CDx * delta
        t17 = t15 * t16
        t18 = t15 * t8
        t19 = t10 * t13
        t20 = t11 * t13
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * ABx * ABz * CDx * CDy * alpha * beta * delta * gamma * t0 ** 19.5 * F0 + 2 * t0 ** 15.5 * t14 * t6 * t7 * F4 - t1 * t4 * t5 - 2 * t1 * (ABx * ABz * CDy * alpha * beta * gamma * t13 + ABz * CDx * CDy * beta * delta * gamma * t13 - t10 * t17 - t11 * t17) - t12 * t5 * (t10 + t11) - 2 * t12 * (-t14 * t4 + t15 * t19 + t15 * t20 - t16 * t18 + t16 * t19 + t16 * t20) - 2 * t13 * t9 * (t16 * t8 + t18 - t19 - t20) - t5 * t8 * t9) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 127:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABx * alpha
        t3 = ABz * beta
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = CDx ** 2 * t1
        t6 = CDx * delta
        t7 = CDx * gamma
        t8 = t0 ** 23.0 * F1
        t9 = q * t8
        t10 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t11 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t10 * t2
        t17 = t11 * t3
        t18 = t16 - t17
        t19 = t10 * t6 - t10 * t7
        t20 = t17 * t2
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDx * t8 * (ABx * ABz * alpha * beta * gamma * t11 + ABz * CDx * beta * delta * gamma * t11 - delta * t20 - gamma * t16 * t6) + p * t3 * t9 * (t2 + t6 - t7) + q * t10 * t13 - q * t4 + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (t18 + t19) + t12 * t15 * (3 * ABz * beta * t11 - t16 - t19) + 2 * t13 * (ABx * CDx * alpha * delta * t10 + ABz * CDx * beta * gamma * t11 - t10 * t5 - t16 * t7 - t17 * t6 - t20) + t18 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 128:
        t0 = q ** (-1.0)
        t1 = p + q
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-2.0) * (-ABy * beta * t1 ** 1.5 * F0 + t1 ** 0.5 * (p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)) * F1) * KAB * KCD / p ** 2
    if case_id == 129:
        t0 = p + q
        t1 = ABy * beta
        t2 = CDz * gamma
        t3 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t4 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        return -2 * np.pi ** 2.5 * t0 ** (-4.5) * (t0 ** 2.0 * t3 * t4 * F2 - t0 ** 3.0 * (t1 * t4 + t2 * t3) * F1 + t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 130:
        t0 = p + q
        t1 = F1
        t2 = t0 ** 4.5
        t3 = ABy * beta
        t4 = CDy * gamma
        t5 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        return np.pi ** 2.5 * t0 ** (-6.0) * (p * q * t1 * t2 - 2 * t0 ** 3.5 * t5 ** 2 * F2 - 2 * t0 ** 5.5 * t3 * t4 * F0 - 2 * t1 * t2 * t5 * (t3 + t4)) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 131:
        t0 = p + q
        t1 = ABy * beta
        t2 = CDx * gamma
        t3 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t4 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        return -2 * np.pi ** 2.5 * t0 ** (-4.5) * (t0 ** 2.0 * t3 * t4 * F2 - t0 ** 3.0 * (t1 * t3 + t2 * t4) * F1 + t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 132:
        t0 = p + q
        t1 = ABy * beta
        t2 = CDz * delta
        t3 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t4 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (-t0 ** 2.0 * t3 * t4 * F2 + t0 ** 3.0 * (t1 * t4 - t2 * t3) * F1 + t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 133:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABy * beta
        t3 = q * t2
        t4 = t0 ** 13.5 * F0
        t5 = t0 ** 12.5 * F1
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = q * t6
        t8 = t0 ** 11.5 * F2
        t9 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t10 = t2 * t9
        t11 = CDz * t6
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * CDz ** 2 * t1 * t2 * t4 + 2 * CDz * t5 * (ABy * beta * gamma * t9 - delta * gamma * t11 - delta * t10) + p * t3 * t5 - p * t7 * t8 + 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - t3 * t4 + t5 * t7 + 2 * t8 * t9 * (CDz * delta * t6 - gamma * t11 - t10)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 134:
        t0 = p + q
        t1 = CDz * delta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABy * beta
        t5 = CDy * gamma
        t6 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 - t1 * t2 * t3 + 2 * t2 * (ABy * CDy * beta * gamma * t6 - t4 * t9 - t5 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (CDz * delta * t8 - t4 * t6 - t5 * t6)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 135:
        t0 = p + q
        t1 = CDx * gamma
        t2 = CDz * delta
        t3 = ABy * beta
        t4 = t2 * t3
        t5 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t8 = t1 * t7
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (CDz * delta * t6 * t7 - t3 * t5 * t6 - t5 * t8) * F2 + t0 ** 6.5 * (ABy * CDx * beta * gamma * t5 - t2 * t8 - t4 * t6) * F1 + t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 136:
        t0 = p + q
        t1 = t0 ** 4.5 * F1
        t2 = ABy * beta
        t3 = CDy * delta
        t4 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        return np.pi ** 2.5 * t0 ** (-6.0) * (p * q * t1 - 2 * t0 ** 3.5 * t4 ** 2 * F2 + 2 * t0 ** 5.5 * t2 * t3 * F0 + 2 * t1 * t4 * (t2 - t3)) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 137:
        t0 = p + q
        t1 = CDz * gamma
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = CDy * delta
        t5 = ABy * beta
        t6 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (ABy * CDz * beta * gamma * t8 - t4 * t5 * t6 - t4 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (CDy * delta * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 138:
        t0 = p + q
        t1 = ABy * beta
        t2 = t0 ** 13.5 * F0
        t3 = -CDy * delta + CDy * gamma + t1
        t4 = t0 ** 12.5 * F1
        t5 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t6 = q * t5
        t7 = t0 ** 11.5 * F2
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * CDy ** 2 * delta * gamma * t1 * t2 + 2 * CDy * t4 * t5 * (ABy * beta * gamma - CDy * delta * gamma - delta * t1) + p * q * t3 * t4 - 3 * p * t6 * t7 - q * t1 * t2 + 2 * t0 ** 10.5 * t5 ** 3 * F3 - 2 * t3 * t5 ** 2 * t7 + t4 * t6) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 139:
        t0 = p + q
        t1 = CDx * gamma
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = CDy * delta
        t5 = ABy * beta
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (ABy * CDx * beta * gamma * t8 - t4 * t5 * t6 - t4 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (CDy * delta * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 140:
        t0 = p + q
        t1 = ABy * beta
        t2 = CDx * delta
        t3 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t4 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (-t0 ** 2.0 * t3 * t4 * F2 + t0 ** 3.0 * (t1 * t3 - t2 * t4) * F1 + t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 141:
        t0 = p + q
        t1 = CDz * gamma
        t2 = CDx * delta
        t3 = ABy * beta
        t4 = t2 * t3
        t5 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t8 = t1 * t6
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (CDx * delta * t6 * t7 - t3 * t5 * t7 - t5 * t8) * F2 + t0 ** 6.5 * (ABy * CDz * beta * gamma * t5 - t2 * t8 - t4 * t7) * F1 + t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 142:
        t0 = p + q
        t1 = CDx * delta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABy * beta
        t5 = CDy * gamma
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 - t1 * t2 * t3 + 2 * t2 * (ABy * CDy * beta * gamma * t6 - t4 * t9 - t5 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (CDx * delta * t8 - t4 * t6 - t5 * t6)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 143:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABy * beta
        t3 = q * t2
        t4 = t0 ** 13.5 * F0
        t5 = t0 ** 12.5 * F1
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = q * t6
        t8 = t0 ** 11.5 * F2
        t9 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t10 = t2 * t9
        t11 = CDx * t6
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * CDx ** 2 * t1 * t2 * t4 + 2 * CDx * t5 * (ABy * beta * gamma * t9 - delta * gamma * t11 - delta * t10) + p * t3 * t5 - p * t7 * t8 + 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - t3 * t4 + t5 * t7 + 2 * t8 * t9 * (CDx * delta * t6 - gamma * t11 - t10)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 144:
        t0 = q ** (-1.0)
        t1 = p + q
        t2 = ABy * beta
        t3 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t4 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-4.5) * (-ABz * alpha * t1 ** 4.0 * t2 * F0 + t1 ** 2.0 * t3 * t4 * F2 - t1 ** 3.0 * (ABz * alpha * t3 - t2 * t4) * F1) * KAB * KCD / p ** 3
    if case_id == 145:
        t0 = p + q
        t1 = ABy * beta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABz * alpha
        t5 = CDz * gamma
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = t0 ** 9.5 * F2
        t8 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 - t1 * t2 * t3 - 2 * t2 * (t4 * t5 * t6 + t4 * t9 - t5 * t9) - t3 * t6 * t7 - 2 * t7 * t8 * (ABz * alpha * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 146:
        t0 = p + q
        t1 = F1
        t2 = t0 ** 10.5
        t3 = ABz * alpha
        t4 = ABy * beta
        t5 = CDy * gamma
        t6 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t7 = t0 ** 9.5 * F2
        t8 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t9 = t3 * t8
        t10 = t4 * t6
        return np.pi ** 2.5 * t0 ** (-12.0) * (ABz * alpha * p * q * t1 * t2 - p * q * t6 * t7 + 2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t3 * t4 * t5 * F0 - 2 * t1 * t2 * (-t10 * t5 + t4 * t9 + t5 * t9) - 2 * t7 * t8 * (ABz * alpha * t8 - t10 - t5 * t6)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 147:
        t0 = p + q
        t1 = ABz * alpha
        t2 = ABy * beta
        t3 = CDx * gamma
        t4 = t2 * t3
        t5 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t8 = t2 * t5
        t9 = t3 * t6
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 - t0 ** 5.5 * (ABz * alpha * t5 * t6 - t7 * t8 - t7 * t9) * F2 - t0 ** 6.5 * (t1 * t8 + t1 * t9 - t4 * t7) * F1 - t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 148:
        t0 = p + q
        t1 = ABy * beta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t5 = t0 ** 9.5 * F2
        t6 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t7 = ABz * alpha
        t8 = t1 * t6
        t9 = CDz * delta
        t10 = t4 * t7
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * ABy * ABz * CDz * alpha * beta * delta * t0 ** 11.5 * F0 + 2 * t0 ** 8.5 * t4 * t6 ** 2 * F3 - t1 * t2 * t3 - 2 * t2 * (-t10 * t9 + t7 * t8 + t8 * t9) - t3 * t4 * t5 - 2 * t5 * t6 * (t10 + t4 * t9 - t8)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 149:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABz * alpha
        t3 = ABy * beta
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = CDz ** 2 * t1
        t6 = CDz * delta
        t7 = CDz * gamma
        t8 = t0 ** 23.0 * F1
        t9 = q * t8
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t11 * t3
        t17 = t10 * t2
        t18 = -t17
        t19 = t16 + t18
        t20 = -t10 * t6 + t10 * t7
        t21 = t16 * t2
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDz * t8 * (ABy * ABz * alpha * beta * gamma * t11 + ABy * CDz * beta * delta * gamma * t11 - delta * t21 - gamma * t17 * t6) + p * t3 * t9 * (t2 + t6 - t7) + q * t10 * t13 - q * t4 + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (-t19 - t20) + t12 * t15 * (3 * t16 + t18 + t20) + 2 * t13 * (ABy * CDz * beta * gamma * t11 + ABz * CDz * alpha * delta * t10 - t10 * t5 - t16 * t6 - t17 * t7 - t21) - t19 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 150:
        t0 = p + q
        t1 = F2
        t2 = t0 ** 20.0
        t3 = ABy * beta
        t4 = CDy * gamma
        t5 = t3 * t4
        t6 = ABz * alpha
        t7 = CDz * delta
        t8 = t6 * t7
        t9 = t0 ** 21.0 * F1
        t10 = p * q
        t11 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t12 = t11 ** 2
        t13 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t14 = t13 ** 2
        t15 = t0 ** 19.0 * F3
        t16 = t11 * t3
        t17 = t11 * t4
        t18 = t13 * t6
        t19 = t13 * t7
        t20 = t1 * t2
        return np.pi ** 2.5 * t0 ** (-22.5) * (2 * ABy * ABz * CDy * CDz * alpha * beta * delta * gamma * t0 ** 22.0 * F0 + 1 / 2 * p ** 2 * q ** 2 * t1 * t2 + 2 * t0 ** 18.0 * t12 * t14 * F4 - t10 * t15 * (t12 + t14) - t10 * t20 * (t16 + t17 - t18 - t19) - t10 * t9 * (t5 + t8) - 2 * t11 * t13 * t15 * (ABz * alpha * t11 + CDz * delta * t11 - t13 * t3 - t13 * t4) - 2 * t20 * (-t12 * t8 - t14 * t5 + t16 * t18 + t16 * t19 + t17 * t18 + t17 * t19) - 2 * t9 * (-t16 * t8 - t17 * t8 + t18 * t5 + t19 * t5)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 151:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABy * beta
        t3 = CDx * gamma
        t4 = t2 * t3
        t5 = p * q
        t6 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t7 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t8 = t6 * t7
        t9 = t0 ** 16.5 * F3
        t10 = t2 * t6
        t11 = t3 * t7
        t12 = t0 ** 17.5 * F2
        t13 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t14 = t13 ** 2
        t15 = ABz * alpha
        t16 = t13 * t4
        t17 = CDz * delta
        t18 = t15 * t17
        t19 = t10 * t13
        t20 = t11 * t13
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * ABy * ABz * CDx * CDz * alpha * beta * delta * gamma * t0 ** 19.5 * F0 + 2 * t0 ** 15.5 * t14 * t6 * t7 * F4 - t1 * t4 * t5 - 2 * t1 * (-t10 * t18 - t11 * t18 + t15 * t16 + t16 * t17) - t12 * t5 * (t10 + t11) - 2 * t12 * (-t14 * t4 - t15 * t17 * t8 + t15 * t19 + t15 * t20 + t17 * t19 + t17 * t20) - 2 * t13 * t9 * (ABz * alpha * t6 * t7 + CDz * delta * t6 * t7 - t19 - t20) - t5 * t8 * t9) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 152:
        t0 = p + q
        t1 = ABz * alpha
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABy * beta
        t5 = CDy * delta
        t6 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (-2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (t4 * t5 * t6 + t4 * t9 - t5 * t9) + t3 * t6 * t7 + 2 * t7 * t8 * (ABy * beta * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 153:
        t0 = p + q
        t1 = t0 ** 20.0 * F2
        t2 = ABy * beta
        t3 = CDy * delta
        t4 = t2 * t3
        t5 = CDz * gamma
        t6 = ABz * alpha
        t7 = t5 * t6
        t8 = t0 ** 21.0 * F1
        t9 = p * q
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = t10 ** 2
        t12 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t13 = t12 ** 2
        t14 = t0 ** 19.0 * F3
        t15 = t10 * t2
        t16 = t12 * t5
        t17 = t12 * t6
        t18 = t10 * t3
        return np.pi ** 2.5 * t0 ** (-22.5) * (1 / 2 * p ** 2 * q ** 2 * t1 + 2 * t0 ** 18.0 * t11 * t13 * F4 + 2 * t0 ** 22.0 * t4 * t7 * F0 + t1 * t9 * (t15 + t16 - t17 - t18) + 2 * t1 * (ABy * CDz * beta * gamma * t10 * t12 + ABz * CDy * alpha * delta * t10 * t12 - t11 * t7 - t13 * t4 - t15 * t17 - t16 * t18) + 2 * t10 * t12 * t14 * (ABz * alpha * t10 + CDy * delta * t12 - t10 * t5 - t12 * t2) - t14 * t9 * (t11 + t13) + t8 * t9 * (t4 + t7) + 2 * t8 * (ABy * ABz * CDz * alpha * beta * gamma * t10 + ABy * CDy * CDz * beta * delta * gamma * t12 - t17 * t4 - t18 * t7)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 154:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABy * beta
        t3 = ABz * alpha
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = CDy ** 2 * t1
        t6 = CDy * gamma
        t7 = CDy * delta
        t8 = t0 ** 23.0 * F1
        t9 = q * t8
        t10 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t11 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t10 * t2
        t17 = -ABz * alpha * t11 + t16
        t18 = t11 * t3
        t19 = t10 * t6 - t10 * t7
        t20 = t18 * t2
        t21 = t18 * t6
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDy * t8 * (ABy * ABz * alpha * beta * gamma * t11 + ABy * CDy * beta * delta * gamma * t10 - delta * t20 - delta * t21) + p * t3 * t9 * (t2 + t6 - t7) + q * t10 * t13 - q * t4 + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (-t17 - t19) + t12 * t15 * (t16 - 3 * t18 + t19) + 2 * t13 * (ABy * CDy * beta * gamma * t10 + ABz * CDy * alpha * delta * t11 - t10 * t5 - t16 * t7 - t20 - t21) - t17 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 155:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = CDx * gamma
        t3 = ABz * alpha
        t4 = t2 * t3
        t5 = p * q
        t6 = CDy * delta
        t7 = ABy * beta
        t8 = t6 * t7
        t9 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = t10 * t9
        t12 = t0 ** 16.5 * F3
        t13 = t10 * t3
        t14 = t0 ** 17.5 * F2
        t15 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t16 = t15 ** 2
        t17 = t15 * t4
        t18 = t2 * t9
        t19 = t15 * t18
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t11 * t16 * F4 + 2 * t0 ** 19.5 * t4 * t8 * F0 + t1 * t4 * t5 + 2 * t1 * (-t13 * t8 - t17 * t6 + t17 * t7 + t18 * t8) - t11 * t12 * t5 + 2 * t12 * t15 * (ABz * alpha * t10 * t15 + CDy * delta * t10 * t9 - t11 * t7 - t19) + t14 * t5 * (CDx * gamma * t9 - t13) + 2 * t14 * (ABy * CDx * beta * gamma * t15 * t9 + ABz * CDy * alpha * delta * t10 * t15 - t11 * t6 * t7 - t13 * t15 * t7 - t16 * t4 - t19 * t6)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 156:
        t0 = p + q
        t1 = ABy * beta
        t2 = ABz * alpha
        t3 = CDx * delta
        t4 = t2 * t3
        t5 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t8 = t2 * t6
        t9 = t3 * t7
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (-t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (ABy * beta * t6 * t7 - t5 * t8 - t5 * t9) * F2 + t0 ** 6.5 * (t1 * t8 + t1 * t9 - t4 * t5) * F1 + t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 157:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABy * beta
        t3 = CDx * delta
        t4 = t2 * t3
        t5 = p * q
        t6 = ABz * alpha
        t7 = CDz * gamma
        t8 = t6 * t7
        t9 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = t10 * t9
        t12 = t0 ** 16.5 * F3
        t13 = t2 * t9
        t14 = t10 * t3
        t15 = t0 ** 17.5 * F2
        t16 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t17 = t16 ** 2
        t18 = t13 * t16
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t11 * t17 * F4 + 2 * t0 ** 19.5 * t4 * t8 * F0 + t1 * t4 * t5 + 2 * t1 * (ABy * ABz * CDz * alpha * beta * gamma * t9 + ABy * CDx * CDz * beta * delta * gamma * t16 - t14 * t8 - t16 * t4 * t6) - t11 * t12 * t5 + 2 * t12 * t16 * (ABz * alpha * t10 * t9 + CDx * delta * t10 * t16 - t11 * t7 - t18) + t15 * t5 * (t13 - t14) + 2 * t15 * (ABy * CDz * beta * gamma * t16 * t9 + ABz * CDx * alpha * delta * t10 * t16 - t11 * t6 * t7 - t14 * t16 * t7 - t17 * t4 - t18 * t6)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 158:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABz * alpha
        t3 = CDx * delta
        t4 = t2 * t3
        t5 = p * q
        t6 = ABy * beta
        t7 = CDy * gamma
        t8 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t9 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t10 = t8 * t9
        t11 = t0 ** 16.5 * F3
        t12 = t2 * t8
        t13 = t3 * t9
        t14 = t0 ** 17.5 * F2
        t15 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t16 = t15 ** 2
        t17 = t15 * t4
        t18 = t12 * t15
        t19 = t13 * t15
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t10 * t16 * F4 + 2 * t0 ** 19.5 * t4 * t6 * t7 * F0 - t1 * t4 * t5 + 2 * t1 * (ABy * ABz * CDy * alpha * beta * gamma * t8 + ABy * CDx * CDy * beta * delta * gamma * t9 - t17 * t6 - t17 * t7) - t10 * t11 * t5 + 2 * t11 * t15 * (ABz * alpha * t15 * t8 + CDx * delta * t15 * t9 - t10 * t6 - t10 * t7) - t14 * t5 * (t12 + t13) + 2 * t14 * (ABy * CDy * beta * gamma * t8 * t9 + ABz * CDx * alpha * delta * t16 - t18 * t6 - t18 * t7 - t19 * t6 - t19 * t7)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 159:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABz * alpha
        t3 = ABy * beta
        t4 = t2 * t3
        t5 = t0 ** 24.0 * t4 * F0
        t6 = t0 ** 23.0 * F1
        t7 = q * t6
        t8 = CDx ** 2
        t9 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t10 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t11 = t10 * t9
        t12 = t0 ** 22.0 * F2
        t13 = q * t12
        t14 = t0 ** 21.0 * F3
        t15 = t10 * t3
        t16 = -ABz * alpha * t9 + t15
        t17 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t18 = t17 ** 2
        t19 = t2 * t9
        t20 = CDx * delta
        t21 = t15 * t17
        t22 = gamma * t11
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDx * t6 * (ABy * ABz * alpha * beta * gamma * t17 + ABy * CDx * beta * delta * gamma * t10 - delta * t17 * t4 - gamma * t19 * t20) - p * q * t11 * t14 + p * t13 * t16 + p * t4 * t7 - q * t5 + 2 * t0 ** 20.0 * t11 * t18 * F4 + 2 * t1 * t5 * t8 + t11 * t13 + 2 * t12 * (ABy * CDx * beta * gamma * t10 * t17 + ABz * CDx * alpha * delta * t17 * t9 - CDx * gamma * t17 * t19 - delta * t22 * t8 - t18 * t4 - t20 * t21) + 2 * t14 * t17 * (ABz * alpha * t17 * t9 + CDx * delta * t10 * t9 - CDx * t22 - t21) - t16 * t7) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 160:
        t0 = q ** (-1.0)
        t1 = p + q
        t2 = alpha * beta
        t3 = F0
        t4 = t1 ** 6.0
        t5 = t1 ** 5.0 * F1
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        return np.pi ** 2.5 * t0 * t1 ** (-6.5) * (-2 * ABy ** 2 * t2 * t3 * t4 - 2 * ABy * t5 * t6 * (alpha - beta) - p * q * t5 + p * t3 * t4 + 2 * t1 ** 4.0 * t6 ** 2 * F2) * KAB * KCD / p ** 3
    if case_id == 161:
        t0 = p + q
        t1 = alpha * beta
        t2 = F0
        t3 = t0 ** 13.5
        t4 = t0 ** 12.5
        t5 = F1
        t6 = t4 * t5
        t7 = CDz * gamma
        t8 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t9 = t0 ** 11.5 * F2
        t10 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t11 = ABy * t8
        t12 = t10 * t7
        return np.pi ** 2.5 * t0 ** (-14.0) * (-2 * ABy ** 2 * t1 * t2 * t3 * t7 - 2 * ABy * t6 * (alpha * beta * t11 + alpha * t12 - beta * t12) + CDz * gamma * p * t2 * t3 - p * q * t6 * t7 - p * q * t8 * t9 + p * t4 * t5 * t8 + 2 * t0 ** 10.5 * t10 ** 2 * t8 * F3 - 2 * t10 * t9 * (ABy * alpha * t8 - beta * t11 - t12)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 162:
        t0 = p + q
        t1 = F0
        t2 = t0 ** 13.5
        t3 = CDy * gamma
        t4 = alpha * t3
        t5 = -ABy * alpha + ABy * beta + t3
        t6 = t0 ** 12.5
        t7 = F1
        t8 = t6 * t7
        t9 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t10 = t0 ** 11.5 * F2
        return np.pi ** 2.5 * t0 ** (-14.0) * (-2 * ABy ** 2 * beta * t1 * t2 * t4 - 2 * ABy * t8 * t9 * (ABy * alpha * beta - beta * t3 + t4) + CDy * gamma * p * t1 * t2 - 3 * p * q * t10 * t9 - p * q * t5 * t8 + p * t6 * t7 * t9 + 2 * t0 ** 10.5 * t9 ** 3 * F3 + 2 * t10 * t5 * t9 ** 2) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 163:
        t0 = p + q
        t1 = alpha * beta
        t2 = F0
        t3 = t0 ** 13.5
        t4 = t0 ** 12.5
        t5 = F1
        t6 = t4 * t5
        t7 = CDx * gamma
        t8 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t9 = t0 ** 11.5 * F2
        t10 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t11 = ABy * t8
        t12 = t10 * t7
        return np.pi ** 2.5 * t0 ** (-14.0) * (-2 * ABy ** 2 * t1 * t2 * t3 * t7 - 2 * ABy * t6 * (alpha * beta * t11 + alpha * t12 - beta * t12) + CDx * gamma * p * t2 * t3 - p * q * t6 * t7 - p * q * t8 * t9 + p * t4 * t5 * t8 + 2 * t0 ** 10.5 * t10 ** 2 * t8 * F3 - 2 * t10 * t9 * (ABy * alpha * t8 - beta * t11 - t12)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 164:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDz * delta
        t3 = p * t2
        t4 = t0 ** 13.5 * F0
        t5 = t0 ** 12.5 * F1
        t6 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t7 = p * t6
        t8 = t0 ** 11.5 * F2
        t9 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t10 = ABy * t6
        t11 = t2 * t9
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * ABy ** 2 * t1 * t2 * t4 + 2 * ABy * t5 * (alpha * beta * t10 - alpha * t11 + beta * t11) + q * t3 * t5 + q * t7 * t8 - 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - t3 * t4 - t5 * t7 + 2 * t8 * t9 * (ABy * beta * t6 - alpha * t10 - t11)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 165:
        t0 = p + q
        t1 = alpha * beta
        t2 = delta * gamma
        t3 = 1 / 2 * p
        t4 = t0 ** 34.5 * F0
        t5 = q * t4
        t6 = q ** 2
        t7 = t0 ** 33.5 * F1
        t8 = 1 / 2 * p ** 2
        t9 = q * t7
        t10 = t0 ** 32.5 * F2
        t11 = ABy ** 2 * t1
        t12 = CDz ** 2 * t2
        t13 = t12 * t4
        t14 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t15 = t14 ** 2
        t16 = p * t10
        t17 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t18 = t17 ** 2
        t19 = ABy * t17
        t20 = CDz * t14
        t21 = t0 ** 31.5 * F3
        t22 = alpha * t19
        t23 = delta * t20
        t24 = ABy * t14
        t25 = CDz * t17
        return np.pi ** 2.5 * t0 ** (-35.0) * (2 * ABy * CDz * t7 * (ABy * alpha * beta * gamma * t14 + CDz * beta * delta * gamma * t17 - alpha * t2 * t25 - delta * t1 * t24) - p * q * t21 * (t15 + t18) - p * t13 + p * t20 * t7 * (delta - gamma) + p * t9 * (t11 + t12) + q * t10 * t18 + q * t16 * (ABy * beta * t17 + CDz * gamma * t14 - t22 - t23) + 2 * t0 ** 30.5 * t15 * t18 * F4 + t10 * t6 * t8 + 2 * t10 * (ABy * CDz * alpha * delta * t14 * t17 + ABy * CDz * beta * gamma * t14 * t17 - beta * t19 * t23 - gamma * t20 * t22 - t11 * t15 - t12 * t18) + 2 * t11 * t13 - t11 * t5 + 2 * t14 * t17 * t21 * (alpha * t24 - beta * t24 + delta * t25 - gamma * t25) + t15 * t16 + t19 * t9 * (alpha - beta) + t3 * t5 - t3 * t6 * t7 - t8 * t9) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 166:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDy * gamma
        t3 = CDz * delta
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = ABy ** 2 * t1
        t6 = ABy * beta
        t7 = ABy * alpha
        t8 = t0 ** 23.0 * F1
        t9 = p * t8
        t10 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t11 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t10 * t2
        t17 = t11 * t3
        t18 = -t10 * t6 + t10 * t7 - t16
        t19 = t17 * t2
        t20 = t17 * t6
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * ABy * t8 * (alpha * t16 * t6 - alpha * t19 - alpha * t20 + beta * t19) + p * t10 * t13 - p * t4 + q * t3 * t9 * (t2 + t6 - t7) + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (t17 + t18) + t12 * t15 * (-3 * t17 - t18) + 2 * t13 * (ABy * CDy * beta * gamma * t10 + ABy * CDz * alpha * delta * t11 - t10 * t5 - t16 * t7 - t19 - t20) + 2 * t4 * t5 + t9 * (CDz * delta * t11 - t16)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 167:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDz * delta
        t3 = CDx * gamma
        t4 = t2 * t3
        t5 = t0 ** 24.0 * t4 * F0
        t6 = t0 ** 23.0 * F1
        t7 = p * t6
        t8 = ABy ** 2
        t9 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t10 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t11 = t10 * t9
        t12 = t0 ** 22.0 * F2
        t13 = p * t12
        t14 = t0 ** 21.0 * F3
        t15 = t10 * t3
        t16 = -CDz * delta * t9 + t15
        t17 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t18 = t17 ** 2
        t19 = ABy * alpha
        t20 = beta * t19
        t21 = t17 * t4
        t22 = t2 * t9
        t23 = t17 * t22
        t24 = beta * t11
        t25 = t15 * t17
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * ABy * t6 * (-alpha * t21 + beta * t21 + t15 * t20 - t20 * t22) - p * q * t11 * t14 - p * t5 + q * t13 * t16 + q * t4 * t7 + 2 * t0 ** 20.0 * t11 * t18 * F4 + 2 * t1 * t5 * t8 + t11 * t13 + 2 * t12 * (ABy * CDx * beta * gamma * t10 * t17 + ABy * CDz * alpha * delta * t17 * t9 - ABy * beta * t23 - alpha * t24 * t8 - t18 * t4 - t19 * t25) + 2 * t14 * t17 * (-ABy * t24 + t11 * t19 + t23 - t25) - t16 * t7) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 168:
        t0 = p + q
        t1 = CDy * delta
        t2 = t0 ** 13.5 * F0
        t3 = beta * t1
        t4 = ABy * alpha - ABy * beta + t1
        t5 = t0 ** 12.5 * F1
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = p * t6
        t8 = t0 ** 11.5 * F2
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * ABy ** 2 * alpha * t2 * t3 + 2 * ABy * t5 * t6 * (ABy * alpha * beta - alpha * t1 + t3) + p * q * t4 * t5 - p * t1 * t2 + 3 * q * t7 * t8 - 2 * t0 ** 10.5 * t6 ** 3 * F3 - 2 * t4 * t6 ** 2 * t8 - t5 * t7) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 169:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDy * delta
        t3 = CDz * gamma
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = ABy ** 2 * t1
        t6 = ABy * alpha
        t7 = ABy * beta
        t8 = t0 ** 23.0 * F1
        t9 = p * t8
        t10 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t11 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t10 * t2
        t17 = t11 * t3
        t18 = t16 - t17
        t19 = t10 * t6 - t10 * t7
        t20 = t17 * t2
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * ABy * t8 * (ABy * CDz * alpha * beta * gamma * t11 + CDy * CDz * beta * delta * gamma * t11 - alpha * t20 - beta * t16 * t6) + p * t10 * t13 - p * t4 + q * t3 * t9 * (t2 + t6 - t7) + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (t18 + t19) + t12 * t15 * (3 * CDz * gamma * t11 - t16 - t19) + 2 * t13 * (ABy * CDy * alpha * delta * t10 + ABy * CDz * beta * gamma * t11 - t10 * t5 - t16 * t7 - t17 * t6 - t20) + t18 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 170:
        t0 = p + q
        t1 = alpha * beta
        t2 = delta * gamma
        t3 = 1 / 2 * p
        t4 = t0 ** 34.5 * F0
        t5 = q * t4
        t6 = q ** 2
        t7 = t0 ** 33.5 * F1
        t8 = p ** 2
        t9 = q * t7
        t10 = t0 ** 32.5 * F2
        t11 = ABy ** 2 * t1
        t12 = CDy ** 2 * t2
        t13 = t12 * t4
        t14 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t15 = t14 ** 2
        t16 = p * t15
        t17 = t10 * t15
        t18 = t0 ** 31.5 * F3
        t19 = ABy * t14
        t20 = CDy * t7
        t21 = p * t14
        t22 = ABy * alpha
        t23 = CDy * gamma
        t24 = CDy * delta
        t25 = ABy * beta
        t26 = -ABy * CDy * alpha * delta - ABy * CDy * beta * gamma + t11 + t12 + t22 * t23 + t24 * t25
        t27 = t22 - t23 + t24 - t25
        return np.pi ** 2.5 * t0 ** (-35.0) * (-p * t13 + p * t26 * t9 - 3 * q * t10 * t21 * t27 - 6 * q * t16 * t18 + q * t17 + 2 * t0 ** 30.5 * t14 ** 4 * F4 + t10 * t16 + 3 / 2 * t10 * t6 * t8 + 2 * t11 * t13 - t11 * t5 + 2 * t14 ** 3 * t18 * t27 - 2 * t17 * t26 + 2 * t19 * t20 * (ABy * alpha * beta * gamma + CDy * beta * delta * gamma - alpha * gamma * t24 - beta * delta * t22) + t19 * t9 * (alpha - beta) + t20 * t21 * (delta - gamma) + t3 * t5 - t3 * t6 * t7 - 1 / 2 * t8 * t9) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 171:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDy * delta
        t3 = CDx * gamma
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = ABy ** 2 * t1
        t6 = ABy * alpha
        t7 = ABy * beta
        t8 = t0 ** 23.0 * F1
        t9 = p * t8
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t11 * t3
        t17 = t10 * t2
        t18 = t10 * t6 - t10 * t7 + t17
        t19 = t16 * t6
        t20 = t16 * t2
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * ABy * t8 * (-alpha * t20 - beta * t17 * t6 + beta * t19 + beta * t20) + p * t10 * t13 - p * t4 + q * t3 * t9 * (t2 + t6 - t7) + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (-t16 + t18) + t12 * t15 * (3 * CDx * gamma * t11 - t18) + 2 * t13 * (ABy * CDx * beta * gamma * t11 + ABy * CDy * alpha * delta * t10 - t10 * t5 - t17 * t7 - t19 - t20) + 2 * t4 * t5 + t9 * (-t16 + t17)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 172:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDx * delta
        t3 = p * t2
        t4 = t0 ** 13.5 * F0
        t5 = t0 ** 12.5 * F1
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = p * t6
        t8 = t0 ** 11.5 * F2
        t9 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t10 = ABy * t6
        t11 = t2 * t9
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * ABy ** 2 * t1 * t2 * t4 + 2 * ABy * t5 * (alpha * beta * t10 - alpha * t11 + beta * t11) + q * t3 * t5 + q * t7 * t8 - 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - t3 * t4 - t5 * t7 + 2 * t8 * t9 * (ABy * beta * t6 - alpha * t10 - t11)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 173:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDx * delta
        t3 = CDz * gamma
        t4 = t2 * t3
        t5 = t0 ** 24.0 * t4 * F0
        t6 = t0 ** 23.0 * F1
        t7 = p * t6
        t8 = ABy ** 2
        t9 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = t10 * t9
        t12 = t0 ** 22.0 * F2
        t13 = p * t12
        t14 = t0 ** 21.0 * F3
        t15 = t2 * t9
        t16 = t10 * t3
        t17 = t15 - t16
        t18 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t19 = t18 ** 2
        t20 = ABy * alpha
        t21 = t15 * t18
        t22 = beta * t11
        t23 = t16 * t18
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * ABy * t6 * (ABy * CDz * alpha * beta * gamma * t10 + CDx * CDz * beta * delta * gamma * t18 - alpha * t18 * t4 - beta * t15 * t20) - p * q * t11 * t14 - p * t5 - q * t13 * t17 + q * t4 * t7 + 2 * t0 ** 20.0 * t11 * t19 * F4 + 2 * t1 * t5 * t8 + t11 * t13 + 2 * t12 * (ABy * CDx * alpha * delta * t18 * t9 + ABy * CDz * beta * gamma * t10 * t18 - ABy * beta * t21 - alpha * t22 * t8 - t19 * t4 - t20 * t23) + 2 * t14 * t18 * (-ABy * t22 + t11 * t20 + t21 - t23) + t17 * t7) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 174:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDy * gamma
        t3 = CDx * delta
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = ABy ** 2 * t1
        t6 = ABy * beta
        t7 = ABy * alpha
        t8 = t0 ** 23.0 * F1
        t9 = p * t8
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t11 * t3
        t17 = t10 * t2
        t18 = -t17
        t19 = t16 + t18
        t20 = -t10 * t6 + t10 * t7
        t21 = t16 * t6
        t22 = t16 * t2
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * ABy * t8 * (ABy * CDy * alpha * beta * gamma * t10 + CDx * CDy * beta * delta * gamma * t11 - alpha * t21 - alpha * t22) + p * t10 * t13 - p * t4 + q * t3 * t9 * (t2 + t6 - t7) + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (t19 + t20) + t12 * t15 * (-3 * t16 - t18 - t20) + 2 * t13 * (ABy * CDx * alpha * delta * t11 + ABy * CDy * beta * gamma * t10 - t10 * t5 - t17 * t7 - t21 - t22) + t19 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 175:
        t0 = p + q
        t1 = alpha * beta
        t2 = delta * gamma
        t3 = 1 / 2 * p
        t4 = t0 ** 34.5 * F0
        t5 = q * t4
        t6 = q ** 2
        t7 = t0 ** 33.5 * F1
        t8 = 1 / 2 * p ** 2
        t9 = q * t7
        t10 = t0 ** 32.5 * F2
        t11 = ABy ** 2 * t1
        t12 = CDx ** 2 * t2
        t13 = t12 * t4
        t14 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t15 = t14 ** 2
        t16 = p * t10
        t17 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t18 = t17 ** 2
        t19 = ABy * t17
        t20 = CDx * t14
        t21 = t0 ** 31.5 * F3
        t22 = alpha * t19
        t23 = delta * t20
        t24 = ABy * t14
        t25 = CDx * t17
        return np.pi ** 2.5 * t0 ** (-35.0) * (2 * ABy * CDx * t7 * (ABy * alpha * beta * gamma * t14 + CDx * beta * delta * gamma * t17 - alpha * t2 * t25 - delta * t1 * t24) - p * q * t21 * (t15 + t18) - p * t13 + p * t20 * t7 * (delta - gamma) + p * t9 * (t11 + t12) + q * t10 * t18 + q * t16 * (ABy * beta * t17 + CDx * gamma * t14 - t22 - t23) + 2 * t0 ** 30.5 * t15 * t18 * F4 + t10 * t6 * t8 + 2 * t10 * (ABy * CDx * alpha * delta * t14 * t17 + ABy * CDx * beta * gamma * t14 * t17 - beta * t19 * t23 - gamma * t20 * t22 - t11 * t15 - t12 * t18) + 2 * t11 * t13 - t11 * t5 + 2 * t14 * t17 * t21 * (alpha * t24 - beta * t24 + delta * t25 - gamma * t25) + t15 * t16 + t19 * t9 * (alpha - beta) + t3 * t5 - t3 * t6 * t7 - t8 * t9) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 176:
        t0 = q ** (-1.0)
        t1 = p + q
        t2 = ABx * alpha
        t3 = ABy * beta
        t4 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t5 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-4.5) * (t1 ** 2.0 * t4 * t5 * F2 - t1 ** 3.0 * (t2 * t5 - t3 * t4) * F1 - t1 ** 4.0 * t2 * t3 * F0) * KAB * KCD / p ** 3
    if case_id == 177:
        t0 = p + q
        t1 = ABx * alpha
        t2 = ABy * beta
        t3 = CDz * gamma
        t4 = t2 * t3
        t5 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t8 = t2 * t7
        t9 = t3 * t6
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 - t0 ** 5.5 * (ABx * alpha * t6 * t7 - t5 * t8 - t5 * t9) * F2 - t0 ** 6.5 * (t1 * t8 + t1 * t9 - t4 * t5) * F1 - t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 178:
        t0 = p + q
        t1 = F1
        t2 = t0 ** 10.5
        t3 = ABx * alpha
        t4 = ABy * beta
        t5 = CDy * gamma
        t6 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t7 = t0 ** 9.5 * F2
        t8 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t9 = t3 * t8
        t10 = t4 * t6
        return np.pi ** 2.5 * t0 ** (-12.0) * (ABx * alpha * p * q * t1 * t2 - p * q * t6 * t7 + 2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t3 * t4 * t5 * F0 - 2 * t1 * t2 * (-t10 * t5 + t4 * t9 + t5 * t9) - 2 * t7 * t8 * (ABx * alpha * t8 - t10 - t5 * t6)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 179:
        t0 = p + q
        t1 = ABy * beta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABx * alpha
        t5 = CDx * gamma
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = t0 ** 9.5 * F2
        t8 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 - t1 * t2 * t3 - 2 * t2 * (t4 * t5 * t6 + t4 * t9 - t5 * t9) - t3 * t6 * t7 - 2 * t7 * t8 * (ABx * alpha * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 180:
        t0 = p + q
        t1 = ABy * beta
        t2 = ABx * alpha
        t3 = CDz * delta
        t4 = t2 * t3
        t5 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t8 = t2 * t7
        t9 = t3 * t6
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (-t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (ABy * beta * t6 * t7 - t5 * t8 - t5 * t9) * F2 + t0 ** 6.5 * (t1 * t8 + t1 * t9 - t4 * t5) * F1 + t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 181:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABx * alpha
        t3 = ABy * beta
        t4 = t2 * t3
        t5 = t0 ** 24.0 * t4 * F0
        t6 = t0 ** 23.0 * F1
        t7 = q * t6
        t8 = CDz ** 2
        t9 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = t10 * t9
        t12 = t0 ** 22.0 * F2
        t13 = q * t12
        t14 = t0 ** 21.0 * F3
        t15 = t2 * t9
        t16 = t10 * t3
        t17 = t15 - t16
        t18 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t19 = t18 ** 2
        t20 = CDz * delta
        t21 = t15 * t18
        t22 = t16 * t18
        t23 = gamma * t11
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDz * t6 * (ABx * ABy * alpha * beta * gamma * t18 + ABy * CDz * beta * delta * gamma * t10 - delta * t18 * t4 - gamma * t15 * t20) - p * q * t11 * t14 - p * t13 * t17 + p * t4 * t7 - q * t5 + 2 * t0 ** 20.0 * t11 * t19 * F4 + 2 * t1 * t5 * t8 + t11 * t13 + 2 * t12 * (ABx * CDz * alpha * delta * t18 * t9 + ABy * CDz * beta * gamma * t10 * t18 - CDz * gamma * t21 - delta * t23 * t8 - t19 * t4 - t20 * t22) + 2 * t14 * t18 * (-CDz * t23 + t11 * t20 + t21 - t22) + t17 * t7) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 182:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABx * alpha
        t3 = CDz * delta
        t4 = t2 * t3
        t5 = p * q
        t6 = ABy * beta
        t7 = CDy * gamma
        t8 = t6 * t7
        t9 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = t10 * t9
        t12 = t0 ** 16.5 * F3
        t13 = t2 * t9
        t14 = t10 * t3
        t15 = t0 ** 17.5 * F2
        t16 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t17 = t16 ** 2
        t18 = t16 * t4
        t19 = t13 * t16
        t20 = t14 * t16
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t11 * t17 * F4 + 2 * t0 ** 19.5 * t4 * t8 * F0 - t1 * t4 * t5 + 2 * t1 * (t13 * t8 + t14 * t8 - t18 * t6 - t18 * t7) - t11 * t12 * t5 + 2 * t12 * t16 * (-t11 * t6 - t11 * t7 + t19 + t20) - t15 * t5 * (t13 + t14) + 2 * t15 * (ABx * CDz * alpha * delta * t17 + ABy * CDy * beta * gamma * t10 * t9 - t19 * t6 - t19 * t7 - t20 * t6 - t20 * t7)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 183:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABy * beta
        t3 = CDz * delta
        t4 = t2 * t3
        t5 = p * q
        t6 = ABx * alpha
        t7 = CDx * gamma
        t8 = t6 * t7
        t9 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = t10 * t9
        t12 = t0 ** 16.5 * F3
        t13 = t2 * t9
        t14 = t10 * t3
        t15 = t0 ** 17.5 * F2
        t16 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t17 = t16 ** 2
        t18 = t16 * t4
        t19 = t11 * t6
        t20 = t14 * t16
        t21 = t13 * t16
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t11 * t17 * F4 + 2 * t0 ** 19.5 * t4 * t8 * F0 + t1 * t4 * t5 + 2 * t1 * (t13 * t8 - t14 * t8 - t18 * t6 + t18 * t7) - t11 * t12 * t5 + 2 * t12 * t16 * (-t11 * t7 + t19 + t20 - t21) + t15 * t5 * (t13 - t14) + 2 * t15 * (ABx * CDz * alpha * delta * t10 * t16 + ABy * CDx * beta * gamma * t16 * t9 - t17 * t4 - t19 * t7 - t20 * t7 - t21 * t6)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 184:
        t0 = p + q
        t1 = ABx * alpha
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABy * beta
        t5 = CDy * delta
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (-2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (t4 * t5 * t6 + t4 * t9 - t5 * t9) + t3 * t6 * t7 + 2 * t7 * t8 * (ABy * beta * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 185:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = CDz * gamma
        t3 = ABx * alpha
        t4 = t2 * t3
        t5 = p * q
        t6 = CDy * delta
        t7 = ABy * beta
        t8 = t6 * t7
        t9 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t10 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t11 = t10 * t9
        t12 = t0 ** 16.5 * F3
        t13 = t10 * t3
        t14 = t0 ** 17.5 * F2
        t15 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t16 = t15 ** 2
        t17 = t13 * t15
        t18 = t11 * t6
        t19 = t15 * t2 * t9
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t11 * t16 * F4 + 2 * t0 ** 19.5 * t4 * t8 * F0 + t1 * t4 * t5 + 2 * t1 * (ABx * ABy * CDz * alpha * beta * gamma * t15 + ABy * CDy * CDz * beta * delta * gamma * t9 - t13 * t8 - t15 * t4 * t6) - t11 * t12 * t5 + 2 * t12 * t15 * (-t11 * t7 + t17 + t18 - t19) + t14 * t5 * (CDz * gamma * t9 - t13) + 2 * t14 * (ABx * CDy * alpha * delta * t10 * t15 + ABy * CDz * beta * gamma * t15 * t9 - t16 * t4 - t17 * t7 - t18 * t7 - t19 * t6)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 186:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABy * beta
        t3 = ABx * alpha
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = CDy ** 2 * t1
        t6 = CDy * gamma
        t7 = CDy * delta
        t8 = t0 ** 23.0 * F1
        t9 = q * t8
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t11 * t3
        t17 = t10 * t2
        t18 = -t17
        t19 = t16 + t18
        t20 = -t10 * t6 + t10 * t7
        t21 = t16 * t2
        t22 = t16 * t6
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDy * t8 * (ABx * ABy * alpha * beta * gamma * t11 + ABy * CDy * beta * delta * gamma * t10 - delta * t21 - delta * t22) + p * t3 * t9 * (t2 + t6 - t7) + q * t10 * t13 - q * t4 + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (t19 + t20) + t12 * t15 * (-3 * t16 - t18 - t20) + 2 * t13 * (ABx * CDy * alpha * delta * t11 + ABy * CDy * beta * gamma * t10 - t10 * t5 - t17 * t7 - t21 - t22) + t19 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 187:
        t0 = p + q
        t1 = t0 ** 20.0 * F2
        t2 = CDx * gamma
        t3 = ABx * alpha
        t4 = t2 * t3
        t5 = ABy * beta
        t6 = CDy * delta
        t7 = t5 * t6
        t8 = t0 ** 21.0 * F1
        t9 = p * q
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = t10 ** 2
        t12 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t13 = t12 ** 2
        t14 = t0 ** 19.0 * F3
        t15 = t10 * t3
        t16 = t12 * t6
        t17 = t12 * t5
        t18 = t10 * t2
        return np.pi ** 2.5 * t0 ** (-22.5) * (1 / 2 * p ** 2 * q ** 2 * t1 + 2 * t0 ** 18.0 * t11 * t13 * F4 + 2 * t0 ** 22.0 * t4 * t7 * F0 + t1 * t9 * (ABy * beta * t12 + CDx * gamma * t10 - t15 - t16) + 2 * t1 * (ABx * CDy * alpha * delta * t10 * t12 + ABy * CDx * beta * gamma * t10 * t12 - t11 * t7 - t13 * t4 - t15 * t17 - t16 * t18) + 2 * t10 * t12 * t14 * (-t10 * t5 + t10 * t6 - t12 * t2 + t12 * t3) - t14 * t9 * (t11 + t13) + t8 * t9 * (t4 + t7) + 2 * t8 * (-t15 * t7 - t16 * t4 + t17 * t4 + t18 * t7)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 188:
        t0 = p + q
        t1 = ABy * beta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t5 = t0 ** 9.5 * F2
        t6 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t7 = ABx * alpha
        t8 = t1 * t6
        t9 = CDx * delta
        t10 = t4 * t7
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * ABx * ABy * CDx * alpha * beta * delta * t0 ** 11.5 * F0 + 2 * t0 ** 8.5 * t4 * t6 ** 2 * F3 - t1 * t2 * t3 - 2 * t2 * (-t10 * t9 + t7 * t8 + t8 * t9) - t3 * t4 * t5 - 2 * t5 * t6 * (t10 + t4 * t9 - t8)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 189:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABy * beta
        t3 = CDz * gamma
        t4 = t2 * t3
        t5 = p * q
        t6 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t7 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t8 = t6 * t7
        t9 = t0 ** 16.5 * F3
        t10 = t2 * t6
        t11 = t3 * t7
        t12 = t0 ** 17.5 * F2
        t13 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t14 = t13 ** 2
        t15 = ABx * alpha
        t16 = CDx * delta
        t17 = t15 * t16
        t18 = t15 * t8
        t19 = t10 * t13
        t20 = t11 * t13
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * ABx * ABy * CDx * CDz * alpha * beta * delta * gamma * t0 ** 19.5 * F0 + 2 * t0 ** 15.5 * t14 * t6 * t7 * F4 - t1 * t4 * t5 - 2 * t1 * (ABx * ABy * CDz * alpha * beta * gamma * t13 + ABy * CDx * CDz * beta * delta * gamma * t13 - t10 * t17 - t11 * t17) - t12 * t5 * (t10 + t11) - 2 * t12 * (-t14 * t4 + t15 * t19 + t15 * t20 - t16 * t18 + t16 * t19 + t16 * t20) - 2 * t13 * t9 * (t16 * t8 + t18 - t19 - t20) - t5 * t8 * t9) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 190:
        t0 = p + q
        t1 = F2
        t2 = t0 ** 20.0
        t3 = ABx * alpha
        t4 = CDx * delta
        t5 = t3 * t4
        t6 = ABy * beta
        t7 = CDy * gamma
        t8 = t6 * t7
        t9 = t0 ** 21.0 * F1
        t10 = p * q
        t11 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t12 = t11 ** 2
        t13 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t14 = t13 ** 2
        t15 = t0 ** 19.0 * F3
        t16 = t11 * t3
        t17 = t11 * t4
        t18 = t1 * t2
        t19 = t13 * t6
        t20 = t13 * t7
        return np.pi ** 2.5 * t0 ** (-22.5) * (2 * ABx * ABy * CDx * CDy * alpha * beta * delta * gamma * t0 ** 22.0 * F0 + 1 / 2 * p ** 2 * q ** 2 * t1 * t2 + 2 * t0 ** 18.0 * t12 * t14 * F4 - t10 * t15 * (t12 + t14) - t10 * t18 * (ABy * beta * t13 + CDy * gamma * t13 - t16 - t17) - t10 * t9 * (t5 + t8) - 2 * t11 * t13 * t15 * (-t11 * t6 - t11 * t7 + t13 * t3 + t13 * t4) - 2 * t18 * (-t12 * t8 - t14 * t5 + t16 * t19 + t16 * t20 + t17 * t19 + t17 * t20) - 2 * t9 * (ABx * ABy * CDy * alpha * beta * gamma * t11 + ABy * CDx * CDy * beta * delta * gamma * t11 - t19 * t5 - t20 * t5)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 191:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABx * alpha
        t3 = ABy * beta
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = CDx ** 2 * t1
        t6 = CDx * delta
        t7 = CDx * gamma
        t8 = t0 ** 23.0 * F1
        t9 = q * t8
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t10 * t2
        t17 = t11 * t3
        t18 = t16 - t17
        t19 = t10 * t6 - t10 * t7
        t20 = t17 * t2
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDx * t8 * (ABx * ABy * alpha * beta * gamma * t11 + ABy * CDx * beta * delta * gamma * t11 - delta * t20 - gamma * t16 * t6) + p * t3 * t9 * (t2 + t6 - t7) + q * t10 * t13 - q * t4 + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (t18 + t19) + t12 * t15 * (3 * ABy * beta * t11 - t16 - t19) + 2 * t13 * (ABx * CDx * alpha * delta * t10 + ABy * CDx * beta * gamma * t11 - t10 * t5 - t16 * t7 - t17 * t6 - t20) + t18 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 192:
        t0 = q ** (-1.0)
        t1 = p + q
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-2.0) * (-ABx * beta * t1 ** 1.5 * F0 + t1 ** 0.5 * (p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)) * F1) * KAB * KCD / p ** 2
    if case_id == 193:
        t0 = p + q
        t1 = ABx * beta
        t2 = CDz * gamma
        t3 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t4 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        return -2 * np.pi ** 2.5 * t0 ** (-4.5) * (t0 ** 2.0 * t3 * t4 * F2 - t0 ** 3.0 * (t1 * t4 + t2 * t3) * F1 + t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 194:
        t0 = p + q
        t1 = ABx * beta
        t2 = CDy * gamma
        t3 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t4 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        return -2 * np.pi ** 2.5 * t0 ** (-4.5) * (t0 ** 2.0 * t3 * t4 * F2 - t0 ** 3.0 * (t1 * t4 + t2 * t3) * F1 + t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 195:
        t0 = p + q
        t1 = F1
        t2 = t0 ** 4.5
        t3 = ABx * beta
        t4 = CDx * gamma
        t5 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        return np.pi ** 2.5 * t0 ** (-6.0) * (p * q * t1 * t2 - 2 * t0 ** 3.5 * t5 ** 2 * F2 - 2 * t0 ** 5.5 * t3 * t4 * F0 - 2 * t1 * t2 * t5 * (t3 + t4)) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 196:
        t0 = p + q
        t1 = ABx * beta
        t2 = CDz * delta
        t3 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t4 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (-t0 ** 2.0 * t3 * t4 * F2 + t0 ** 3.0 * (t1 * t4 - t2 * t3) * F1 + t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 197:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABx * beta
        t3 = q * t2
        t4 = t0 ** 13.5 * F0
        t5 = t0 ** 12.5 * F1
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = q * t6
        t8 = t0 ** 11.5 * F2
        t9 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t10 = t2 * t9
        t11 = CDz * t6
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * CDz ** 2 * t1 * t2 * t4 + 2 * CDz * t5 * (ABx * beta * gamma * t9 - delta * gamma * t11 - delta * t10) + p * t3 * t5 - p * t7 * t8 + 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - t3 * t4 + t5 * t7 + 2 * t8 * t9 * (CDz * delta * t6 - gamma * t11 - t10)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 198:
        t0 = p + q
        t1 = CDy * gamma
        t2 = CDz * delta
        t3 = ABx * beta
        t4 = t2 * t3
        t5 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t8 = t1 * t6
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (CDz * delta * t6 * t7 - t3 * t5 * t7 - t5 * t8) * F2 + t0 ** 6.5 * (ABx * CDy * beta * gamma * t5 - t2 * t8 - t4 * t7) * F1 + t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 199:
        t0 = p + q
        t1 = CDz * delta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABx * beta
        t5 = CDx * gamma
        t6 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 - t1 * t2 * t3 + 2 * t2 * (ABx * CDx * beta * gamma * t6 - t4 * t9 - t5 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (CDz * delta * t8 - t4 * t6 - t5 * t6)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 200:
        t0 = p + q
        t1 = ABx * beta
        t2 = CDy * delta
        t3 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t4 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        return 2 * np.pi ** 2.5 * t0 ** (-4.5) * (-t0 ** 2.0 * t3 * t4 * F2 + t0 ** 3.0 * (t1 * t4 - t2 * t3) * F1 + t0 ** 4.0 * t1 * t2 * F0) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 201:
        t0 = p + q
        t1 = CDz * gamma
        t2 = CDy * delta
        t3 = ABx * beta
        t4 = t2 * t3
        t5 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t8 = t1 * t6
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (CDy * delta * t6 * t7 - t3 * t5 * t7 - t5 * t8) * F2 + t0 ** 6.5 * (ABx * CDz * beta * gamma * t5 - t2 * t8 - t4 * t7) * F1 + t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 202:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABx * beta
        t3 = q * t2
        t4 = t0 ** 13.5 * F0
        t5 = t0 ** 12.5 * F1
        t6 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t7 = q * t6
        t8 = t0 ** 11.5 * F2
        t9 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t10 = t2 * t9
        t11 = CDy * t6
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * CDy ** 2 * t1 * t2 * t4 + 2 * CDy * t5 * (ABx * beta * gamma * t9 - delta * gamma * t11 - delta * t10) + p * t3 * t5 - p * t7 * t8 + 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - t3 * t4 + t5 * t7 + 2 * t8 * t9 * (CDy * delta * t6 - gamma * t11 - t10)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 203:
        t0 = p + q
        t1 = CDy * delta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABx * beta
        t5 = CDx * gamma
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 - t1 * t2 * t3 + 2 * t2 * (ABx * CDx * beta * gamma * t6 - t4 * t9 - t5 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (CDy * delta * t8 - t4 * t6 - t5 * t6)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 204:
        t0 = p + q
        t1 = t0 ** 4.5 * F1
        t2 = ABx * beta
        t3 = CDx * delta
        t4 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        return np.pi ** 2.5 * t0 ** (-6.0) * (p * q * t1 - 2 * t0 ** 3.5 * t4 ** 2 * F2 + 2 * t0 ** 5.5 * t2 * t3 * F0 + 2 * t1 * t4 * (t2 - t3)) * KAB * KCD / (p ** 2 * q ** 2)
    if case_id == 205:
        t0 = p + q
        t1 = CDz * gamma
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = CDx * delta
        t5 = ABx * beta
        t6 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (ABx * CDz * beta * gamma * t8 - t4 * t5 * t6 - t4 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (CDx * delta * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 206:
        t0 = p + q
        t1 = CDy * gamma
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = CDx * delta
        t5 = ABx * beta
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (ABx * CDy * beta * gamma * t8 - t4 * t5 * t6 - t4 * t9) - t3 * t6 * t7 + 2 * t7 * t8 * (CDx * delta * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 207:
        t0 = p + q
        t1 = ABx * beta
        t2 = t0 ** 13.5 * F0
        t3 = -CDx * delta + CDx * gamma + t1
        t4 = t0 ** 12.5 * F1
        t5 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t6 = q * t5
        t7 = t0 ** 11.5 * F2
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * CDx ** 2 * delta * gamma * t1 * t2 + 2 * CDx * t4 * t5 * (ABx * beta * gamma - CDx * delta * gamma - delta * t1) + p * q * t3 * t4 - 3 * p * t6 * t7 - q * t1 * t2 + 2 * t0 ** 10.5 * t5 ** 3 * F3 - 2 * t3 * t5 ** 2 * t7 + t4 * t6) * KAB * KCD / (p ** 2 * q ** 3)
    if case_id == 208:
        t0 = q ** (-1.0)
        t1 = p + q
        t2 = ABx * beta
        t3 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t4 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-4.5) * (-ABz * alpha * t1 ** 4.0 * t2 * F0 + t1 ** 2.0 * t3 * t4 * F2 - t1 ** 3.0 * (ABz * alpha * t3 - t2 * t4) * F1) * KAB * KCD / p ** 3
    if case_id == 209:
        t0 = p + q
        t1 = ABx * beta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABz * alpha
        t5 = CDz * gamma
        t6 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t7 = t0 ** 9.5 * F2
        t8 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 - t1 * t2 * t3 - 2 * t2 * (t4 * t5 * t6 + t4 * t9 - t5 * t9) - t3 * t6 * t7 - 2 * t7 * t8 * (ABz * alpha * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 210:
        t0 = p + q
        t1 = ABz * alpha
        t2 = ABx * beta
        t3 = CDy * gamma
        t4 = t2 * t3
        t5 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t8 = t2 * t6
        t9 = t3 * t5
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 - t0 ** 5.5 * (ABz * alpha * t5 * t6 - t7 * t8 - t7 * t9) * F2 - t0 ** 6.5 * (t1 * t8 + t1 * t9 - t4 * t7) * F1 - t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 211:
        t0 = p + q
        t1 = F1
        t2 = t0 ** 10.5
        t3 = ABz * alpha
        t4 = ABx * beta
        t5 = CDx * gamma
        t6 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t7 = t0 ** 9.5 * F2
        t8 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t9 = t3 * t8
        t10 = t4 * t6
        return np.pi ** 2.5 * t0 ** (-12.0) * (ABz * alpha * p * q * t1 * t2 - p * q * t6 * t7 + 2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t3 * t4 * t5 * F0 - 2 * t1 * t2 * (-t10 * t5 + t4 * t9 + t5 * t9) - 2 * t7 * t8 * (ABz * alpha * t8 - t10 - t5 * t6)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 212:
        t0 = p + q
        t1 = ABx * beta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t5 = t0 ** 9.5 * F2
        t6 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t7 = ABz * alpha
        t8 = t1 * t6
        t9 = CDz * delta
        t10 = t4 * t7
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * ABx * ABz * CDz * alpha * beta * delta * t0 ** 11.5 * F0 + 2 * t0 ** 8.5 * t4 * t6 ** 2 * F3 - t1 * t2 * t3 - 2 * t2 * (-t10 * t9 + t7 * t8 + t8 * t9) - t3 * t4 * t5 - 2 * t5 * t6 * (t10 + t4 * t9 - t8)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 213:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABz * alpha
        t3 = ABx * beta
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = CDz ** 2 * t1
        t6 = CDz * delta
        t7 = CDz * gamma
        t8 = t0 ** 23.0 * F1
        t9 = q * t8
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t11 * t3
        t17 = t10 * t2
        t18 = -t17
        t19 = t16 + t18
        t20 = -t10 * t6 + t10 * t7
        t21 = t16 * t2
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDz * t8 * (ABx * ABz * alpha * beta * gamma * t11 + ABx * CDz * beta * delta * gamma * t11 - delta * t21 - gamma * t17 * t6) + p * t3 * t9 * (t2 + t6 - t7) + q * t10 * t13 - q * t4 + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (-t19 - t20) + t12 * t15 * (3 * t16 + t18 + t20) + 2 * t13 * (ABx * CDz * beta * gamma * t11 + ABz * CDz * alpha * delta * t10 - t10 * t5 - t16 * t6 - t17 * t7 - t21) - t19 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 214:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABx * beta
        t3 = CDy * gamma
        t4 = t2 * t3
        t5 = p * q
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t8 = t6 * t7
        t9 = t0 ** 16.5 * F3
        t10 = t2 * t6
        t11 = t3 * t7
        t12 = t0 ** 17.5 * F2
        t13 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t14 = t13 ** 2
        t15 = ABz * alpha
        t16 = t13 * t4
        t17 = CDz * delta
        t18 = t15 * t17
        t19 = t10 * t13
        t20 = t11 * t13
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * ABx * ABz * CDy * CDz * alpha * beta * delta * gamma * t0 ** 19.5 * F0 + 2 * t0 ** 15.5 * t14 * t6 * t7 * F4 - t1 * t4 * t5 - 2 * t1 * (-t10 * t18 - t11 * t18 + t15 * t16 + t16 * t17) - t12 * t5 * (t10 + t11) - 2 * t12 * (-t14 * t4 - t15 * t17 * t8 + t15 * t19 + t15 * t20 + t17 * t19 + t17 * t20) - 2 * t13 * t9 * (ABz * alpha * t6 * t7 + CDz * delta * t6 * t7 - t19 - t20) - t5 * t8 * t9) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 215:
        t0 = p + q
        t1 = F2
        t2 = t0 ** 20.0
        t3 = ABx * beta
        t4 = CDx * gamma
        t5 = t3 * t4
        t6 = ABz * alpha
        t7 = CDz * delta
        t8 = t6 * t7
        t9 = t0 ** 21.0 * F1
        t10 = p * q
        t11 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t12 = t11 ** 2
        t13 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t14 = t13 ** 2
        t15 = t0 ** 19.0 * F3
        t16 = t11 * t3
        t17 = t11 * t4
        t18 = t13 * t6
        t19 = t13 * t7
        t20 = t1 * t2
        return np.pi ** 2.5 * t0 ** (-22.5) * (2 * ABx * ABz * CDx * CDz * alpha * beta * delta * gamma * t0 ** 22.0 * F0 + 1 / 2 * p ** 2 * q ** 2 * t1 * t2 + 2 * t0 ** 18.0 * t12 * t14 * F4 - t10 * t15 * (t12 + t14) - t10 * t20 * (t16 + t17 - t18 - t19) - t10 * t9 * (t5 + t8) - 2 * t11 * t13 * t15 * (ABz * alpha * t11 + CDz * delta * t11 - t13 * t3 - t13 * t4) - 2 * t20 * (-t12 * t8 - t14 * t5 + t16 * t18 + t16 * t19 + t17 * t18 + t17 * t19) - 2 * t9 * (-t16 * t8 - t17 * t8 + t18 * t5 + t19 * t5)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 216:
        t0 = p + q
        t1 = ABx * beta
        t2 = ABz * alpha
        t3 = CDy * delta
        t4 = t2 * t3
        t5 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t8 = t2 * t6
        t9 = t3 * t7
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (-t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (ABx * beta * t6 * t7 - t5 * t8 - t5 * t9) * F2 + t0 ** 6.5 * (t1 * t8 + t1 * t9 - t4 * t5) * F1 + t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 217:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABx * beta
        t3 = CDy * delta
        t4 = t2 * t3
        t5 = p * q
        t6 = ABz * alpha
        t7 = CDz * gamma
        t8 = t6 * t7
        t9 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = t10 * t9
        t12 = t0 ** 16.5 * F3
        t13 = t2 * t9
        t14 = t10 * t3
        t15 = t0 ** 17.5 * F2
        t16 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t17 = t16 ** 2
        t18 = t13 * t16
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t11 * t17 * F4 + 2 * t0 ** 19.5 * t4 * t8 * F0 + t1 * t4 * t5 + 2 * t1 * (ABx * ABz * CDz * alpha * beta * gamma * t9 + ABx * CDy * CDz * beta * delta * gamma * t16 - t14 * t8 - t16 * t4 * t6) - t11 * t12 * t5 + 2 * t12 * t16 * (ABz * alpha * t10 * t9 + CDy * delta * t10 * t16 - t11 * t7 - t18) + t15 * t5 * (t13 - t14) + 2 * t15 * (ABx * CDz * beta * gamma * t16 * t9 + ABz * CDy * alpha * delta * t10 * t16 - t11 * t6 * t7 - t14 * t16 * t7 - t17 * t4 - t18 * t6)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 218:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABz * alpha
        t3 = ABx * beta
        t4 = t2 * t3
        t5 = t0 ** 24.0 * t4 * F0
        t6 = t0 ** 23.0 * F1
        t7 = q * t6
        t8 = CDy ** 2
        t9 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t10 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t11 = t10 * t9
        t12 = t0 ** 22.0 * F2
        t13 = q * t12
        t14 = t0 ** 21.0 * F3
        t15 = t10 * t3
        t16 = -ABz * alpha * t9 + t15
        t17 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t18 = t17 ** 2
        t19 = t2 * t9
        t20 = CDy * delta
        t21 = t15 * t17
        t22 = gamma * t11
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDy * t6 * (ABx * ABz * alpha * beta * gamma * t17 + ABx * CDy * beta * delta * gamma * t10 - delta * t17 * t4 - gamma * t19 * t20) - p * q * t11 * t14 + p * t13 * t16 + p * t4 * t7 - q * t5 + 2 * t0 ** 20.0 * t11 * t18 * F4 + 2 * t1 * t5 * t8 + t11 * t13 + 2 * t12 * (ABx * CDy * beta * gamma * t10 * t17 + ABz * CDy * alpha * delta * t17 * t9 - CDy * gamma * t17 * t19 - delta * t22 * t8 - t18 * t4 - t20 * t21) + 2 * t14 * t17 * (ABz * alpha * t17 * t9 + CDy * delta * t10 * t9 - CDy * t22 - t21) - t16 * t7) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 219:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABz * alpha
        t3 = CDy * delta
        t4 = t2 * t3
        t5 = p * q
        t6 = ABx * beta
        t7 = CDx * gamma
        t8 = t6 * t7
        t9 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t10 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t11 = t10 * t9
        t12 = t0 ** 16.5 * F3
        t13 = t2 * t9
        t14 = t10 * t3
        t15 = t0 ** 17.5 * F2
        t16 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t17 = t16 ** 2
        t18 = t16 * t4
        t19 = t13 * t16
        t20 = t14 * t16
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t11 * t17 * F4 + 2 * t0 ** 19.5 * t4 * t8 * F0 - t1 * t4 * t5 + 2 * t1 * (t13 * t8 + t14 * t8 - t18 * t6 - t18 * t7) - t11 * t12 * t5 + 2 * t12 * t16 * (ABz * alpha * t16 * t9 + CDy * delta * t10 * t16 - t11 * t6 - t11 * t7) - t15 * t5 * (t13 + t14) + 2 * t15 * (ABx * CDx * beta * gamma * t10 * t9 + ABz * CDy * alpha * delta * t17 - t19 * t6 - t19 * t7 - t20 * t6 - t20 * t7)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 220:
        t0 = p + q
        t1 = ABz * alpha
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABx * beta
        t5 = CDx * delta
        t6 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (-2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (t4 * t5 * t6 + t4 * t9 - t5 * t9) + t3 * t6 * t7 + 2 * t7 * t8 * (ABx * beta * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 221:
        t0 = p + q
        t1 = t0 ** 20.0 * F2
        t2 = ABx * beta
        t3 = CDx * delta
        t4 = t2 * t3
        t5 = CDz * gamma
        t6 = ABz * alpha
        t7 = t5 * t6
        t8 = t0 ** 21.0 * F1
        t9 = p * q
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = t10 ** 2
        t12 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t13 = t12 ** 2
        t14 = t0 ** 19.0 * F3
        t15 = t10 * t2
        t16 = t12 * t5
        t17 = t12 * t6
        t18 = t10 * t3
        return np.pi ** 2.5 * t0 ** (-22.5) * (1 / 2 * p ** 2 * q ** 2 * t1 + 2 * t0 ** 18.0 * t11 * t13 * F4 + 2 * t0 ** 22.0 * t4 * t7 * F0 + t1 * t9 * (t15 + t16 - t17 - t18) + 2 * t1 * (ABx * CDz * beta * gamma * t10 * t12 + ABz * CDx * alpha * delta * t10 * t12 - t11 * t7 - t13 * t4 - t15 * t17 - t16 * t18) + 2 * t10 * t12 * t14 * (ABz * alpha * t10 + CDx * delta * t12 - t10 * t5 - t12 * t2) - t14 * t9 * (t11 + t13) + t8 * t9 * (t4 + t7) + 2 * t8 * (ABx * ABz * CDz * alpha * beta * gamma * t10 + ABx * CDx * CDz * beta * delta * gamma * t12 - t17 * t4 - t18 * t7)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 222:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = CDy * gamma
        t3 = ABz * alpha
        t4 = t2 * t3
        t5 = p * q
        t6 = CDx * delta
        t7 = ABx * beta
        t8 = t6 * t7
        t9 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = t10 * t9
        t12 = t0 ** 16.5 * F3
        t13 = t10 * t3
        t14 = t0 ** 17.5 * F2
        t15 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t16 = t15 ** 2
        t17 = t15 * t2 * t9
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t11 * t16 * F4 + 2 * t0 ** 19.5 * t4 * t8 * F0 + t1 * t4 * t5 + 2 * t1 * (ABx * ABz * CDy * alpha * beta * gamma * t15 + ABx * CDx * CDy * beta * delta * gamma * t9 - t13 * t8 - t15 * t4 * t6) - t11 * t12 * t5 + 2 * t12 * t15 * (ABz * alpha * t10 * t15 + CDx * delta * t10 * t9 - t11 * t7 - t17) + t14 * t5 * (CDy * gamma * t9 - t13) + 2 * t14 * (ABx * CDy * beta * gamma * t15 * t9 + ABz * CDx * alpha * delta * t10 * t15 - t11 * t6 * t7 - t13 * t15 * t7 - t16 * t4 - t17 * t6)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 223:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABx * beta
        t3 = ABz * alpha
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = CDx ** 2 * t1
        t6 = CDx * gamma
        t7 = CDx * delta
        t8 = t0 ** 23.0 * F1
        t9 = q * t8
        t10 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t11 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t10 * t2
        t17 = -ABz * alpha * t11 + t16
        t18 = t11 * t3
        t19 = t10 * t6 - t10 * t7
        t20 = t18 * t2
        t21 = t18 * t6
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDx * t8 * (ABx * ABz * alpha * beta * gamma * t11 + ABx * CDx * beta * delta * gamma * t10 - delta * t20 - delta * t21) + p * t3 * t9 * (t2 + t6 - t7) + q * t10 * t13 - q * t4 + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (-t17 - t19) + t12 * t15 * (t16 - 3 * t18 + t19) + 2 * t13 * (ABx * CDx * beta * gamma * t10 + ABz * CDx * alpha * delta * t11 - t10 * t5 - t16 * t7 - t20 - t21) - t17 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 224:
        t0 = q ** (-1.0)
        t1 = p + q
        t2 = ABx * beta
        t3 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t4 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        return 2 * np.pi ** 2.5 * t0 * t1 ** (-4.5) * (-ABy * alpha * t1 ** 4.0 * t2 * F0 + t1 ** 2.0 * t3 * t4 * F2 - t1 ** 3.0 * (ABy * alpha * t3 - t2 * t4) * F1) * KAB * KCD / p ** 3
    if case_id == 225:
        t0 = p + q
        t1 = ABy * alpha
        t2 = ABx * beta
        t3 = CDz * gamma
        t4 = t2 * t3
        t5 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t8 = t2 * t7
        t9 = t3 * t5
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (t0 ** 4.5 * t5 * t6 * t7 * F3 - t0 ** 5.5 * (ABy * alpha * t5 * t7 - t6 * t8 - t6 * t9) * F2 - t0 ** 6.5 * (t1 * t8 + t1 * t9 - t4 * t6) * F1 - t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 226:
        t0 = p + q
        t1 = ABx * beta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABy * alpha
        t5 = CDy * gamma
        t6 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t7 = t0 ** 9.5 * F2
        t8 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 - t1 * t2 * t3 - 2 * t2 * (t4 * t5 * t6 + t4 * t9 - t5 * t9) - t3 * t6 * t7 - 2 * t7 * t8 * (ABy * alpha * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 227:
        t0 = p + q
        t1 = F1
        t2 = t0 ** 10.5
        t3 = ABy * alpha
        t4 = ABx * beta
        t5 = CDx * gamma
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = t0 ** 9.5 * F2
        t8 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t9 = t3 * t8
        t10 = t4 * t6
        return np.pi ** 2.5 * t0 ** (-12.0) * (ABy * alpha * p * q * t1 * t2 - p * q * t6 * t7 + 2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 - 2 * t0 ** 11.5 * t3 * t4 * t5 * F0 - 2 * t1 * t2 * (-t10 * t5 + t4 * t9 + t5 * t9) - 2 * t7 * t8 * (ABy * alpha * t8 - t10 - t5 * t6)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 228:
        t0 = p + q
        t1 = ABx * beta
        t2 = ABy * alpha
        t3 = CDz * delta
        t4 = t2 * t3
        t5 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t8 = t2 * t7
        t9 = t3 * t6
        return 2 * np.pi ** 2.5 * t0 ** (-8.0) * (-t0 ** 4.5 * t5 * t6 * t7 * F3 + t0 ** 5.5 * (ABx * beta * t6 * t7 - t5 * t8 - t5 * t9) * F2 + t0 ** 6.5 * (t1 * t8 + t1 * t9 - t4 * t5) * F1 + t0 ** 7.5 * t1 * t4 * F0) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 229:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABy * alpha
        t3 = ABx * beta
        t4 = t2 * t3
        t5 = t0 ** 24.0 * t4 * F0
        t6 = t0 ** 23.0 * F1
        t7 = q * t6
        t8 = CDz ** 2
        t9 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = t10 * t9
        t12 = t0 ** 22.0 * F2
        t13 = q * t12
        t14 = t0 ** 21.0 * F3
        t15 = t10 * t3
        t16 = -ABy * alpha * t9 + t15
        t17 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t18 = t17 ** 2
        t19 = t2 * t9
        t20 = CDz * delta
        t21 = t15 * t17
        t22 = gamma * t11
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDz * t6 * (ABx * ABy * alpha * beta * gamma * t17 + ABx * CDz * beta * delta * gamma * t10 - delta * t17 * t4 - gamma * t19 * t20) - p * q * t11 * t14 + p * t13 * t16 + p * t4 * t7 - q * t5 + 2 * t0 ** 20.0 * t11 * t18 * F4 + 2 * t1 * t5 * t8 + t11 * t13 + 2 * t12 * (ABx * CDz * beta * gamma * t10 * t17 + ABy * CDz * alpha * delta * t17 * t9 - CDz * gamma * t17 * t19 - delta * t22 * t8 - t18 * t4 - t20 * t21) + 2 * t14 * t17 * (ABy * alpha * t17 * t9 + CDz * delta * t10 * t9 - CDz * t22 - t21) - t16 * t7) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 230:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABx * beta
        t3 = CDz * delta
        t4 = t2 * t3
        t5 = p * q
        t6 = ABy * alpha
        t7 = CDy * gamma
        t8 = t6 * t7
        t9 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = t10 * t9
        t12 = t0 ** 16.5 * F3
        t13 = t2 * t9
        t14 = t10 * t3
        t15 = t0 ** 17.5 * F2
        t16 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t17 = t16 ** 2
        t18 = t16 * t4
        t19 = t13 * t16
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t11 * t17 * F4 + 2 * t0 ** 19.5 * t4 * t8 * F0 + t1 * t4 * t5 + 2 * t1 * (t13 * t8 - t14 * t8 - t18 * t6 + t18 * t7) - t11 * t12 * t5 + 2 * t12 * t16 * (ABy * alpha * t10 * t9 + CDz * delta * t10 * t16 - t11 * t7 - t19) + t15 * t5 * (t13 - t14) + 2 * t15 * (ABx * CDy * beta * gamma * t16 * t9 + ABy * CDz * alpha * delta * t10 * t16 - t11 * t6 * t7 - t14 * t16 * t7 - t17 * t4 - t19 * t6)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 231:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABy * alpha
        t3 = CDz * delta
        t4 = t2 * t3
        t5 = p * q
        t6 = ABx * beta
        t7 = CDx * gamma
        t8 = t6 * t7
        t9 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = t10 * t9
        t12 = t0 ** 16.5 * F3
        t13 = t2 * t9
        t14 = t10 * t3
        t15 = t0 ** 17.5 * F2
        t16 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t17 = t16 ** 2
        t18 = t16 * t4
        t19 = t13 * t16
        t20 = t14 * t16
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t11 * t17 * F4 + 2 * t0 ** 19.5 * t4 * t8 * F0 - t1 * t4 * t5 + 2 * t1 * (t13 * t8 + t14 * t8 - t18 * t6 - t18 * t7) - t11 * t12 * t5 + 2 * t12 * t16 * (ABy * alpha * t16 * t9 + CDz * delta * t10 * t16 - t11 * t6 - t11 * t7) - t15 * t5 * (t13 + t14) + 2 * t15 * (ABx * CDx * beta * gamma * t10 * t9 + ABy * CDz * alpha * delta * t17 - t19 * t6 - t19 * t7 - t20 * t6 - t20 * t7)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 232:
        t0 = p + q
        t1 = ABx * beta
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t5 = t0 ** 9.5 * F2
        t6 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t7 = ABy * alpha
        t8 = t1 * t6
        t9 = CDy * delta
        t10 = t4 * t7
        return np.pi ** 2.5 * t0 ** (-12.0) * (2 * ABx * ABy * CDy * alpha * beta * delta * t0 ** 11.5 * F0 + 2 * t0 ** 8.5 * t4 * t6 ** 2 * F3 - t1 * t2 * t3 - 2 * t2 * (-t10 * t9 + t7 * t8 + t8 * t9) - t3 * t4 * t5 - 2 * t5 * t6 * (t10 + t4 * t9 - t8)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 233:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = ABx * beta
        t3 = CDz * gamma
        t4 = t2 * t3
        t5 = p * q
        t6 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t7 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t8 = t6 * t7
        t9 = t0 ** 16.5 * F3
        t10 = t2 * t6
        t11 = t3 * t7
        t12 = t0 ** 17.5 * F2
        t13 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t14 = t13 ** 2
        t15 = ABy * alpha
        t16 = CDy * delta
        t17 = t15 * t16
        t18 = t10 * t13
        t19 = t11 * t13
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * ABx * ABy * CDy * CDz * alpha * beta * delta * gamma * t0 ** 19.5 * F0 + 2 * t0 ** 15.5 * t14 * t6 * t7 * F4 - t1 * t4 * t5 - 2 * t1 * (ABx * ABy * CDz * alpha * beta * gamma * t13 + ABx * CDy * CDz * beta * delta * gamma * t13 - t10 * t17 - t11 * t17) - t12 * t5 * (t10 + t11) - 2 * t12 * (-t14 * t4 - t15 * t16 * t8 + t15 * t18 + t15 * t19 + t16 * t18 + t16 * t19) - 2 * t13 * t9 * (ABy * alpha * t6 * t7 + CDy * delta * t6 * t7 - t18 - t19) - t5 * t8 * t9) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 234:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABy * alpha
        t3 = ABx * beta
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = CDy ** 2 * t1
        t6 = CDy * delta
        t7 = CDy * gamma
        t8 = t0 ** 23.0 * F1
        t9 = q * t8
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t11 * t3
        t17 = t10 * t2
        t18 = -t17
        t19 = t16 + t18
        t20 = -t10 * t6 + t10 * t7
        t21 = t16 * t2
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDy * t8 * (ABx * ABy * alpha * beta * gamma * t11 + ABx * CDy * beta * delta * gamma * t11 - delta * t21 - gamma * t17 * t6) + p * t3 * t9 * (t2 + t6 - t7) + q * t10 * t13 - q * t4 + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (-t19 - t20) + t12 * t15 * (3 * t16 + t18 + t20) + 2 * t13 * (ABx * CDy * beta * gamma * t11 + ABy * CDy * alpha * delta * t10 - t10 * t5 - t16 * t6 - t17 * t7 - t21) - t19 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 235:
        t0 = p + q
        t1 = F2
        t2 = t0 ** 20.0
        t3 = ABx * beta
        t4 = CDx * gamma
        t5 = t3 * t4
        t6 = ABy * alpha
        t7 = CDy * delta
        t8 = t6 * t7
        t9 = t0 ** 21.0 * F1
        t10 = p * q
        t11 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t12 = t11 ** 2
        t13 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t14 = t13 ** 2
        t15 = t0 ** 19.0 * F3
        t16 = t11 * t3
        t17 = t11 * t4
        t18 = t13 * t6
        t19 = t13 * t7
        t20 = t1 * t2
        return np.pi ** 2.5 * t0 ** (-22.5) * (2 * ABx * ABy * CDx * CDy * alpha * beta * delta * gamma * t0 ** 22.0 * F0 + 1 / 2 * p ** 2 * q ** 2 * t1 * t2 + 2 * t0 ** 18.0 * t12 * t14 * F4 - t10 * t15 * (t12 + t14) - t10 * t20 * (t16 + t17 - t18 - t19) - t10 * t9 * (t5 + t8) - 2 * t11 * t13 * t15 * (ABy * alpha * t11 + CDy * delta * t11 - t13 * t3 - t13 * t4) - 2 * t20 * (-t12 * t8 - t14 * t5 + t16 * t18 + t16 * t19 + t17 * t18 + t17 * t19) - 2 * t9 * (-t16 * t8 - t17 * t8 + t18 * t5 + t19 * t5)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 236:
        t0 = p + q
        t1 = ABy * alpha
        t2 = t0 ** 10.5 * F1
        t3 = p * q
        t4 = ABx * beta
        t5 = CDx * delta
        t6 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t7 = t0 ** 9.5 * F2
        t8 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t9 = t1 * t8
        return np.pi ** 2.5 * t0 ** (-12.0) * (-2 * t0 ** 8.5 * t6 * t8 ** 2 * F3 + 2 * t0 ** 11.5 * t1 * t4 * t5 * F0 + t1 * t2 * t3 + 2 * t2 * (t4 * t5 * t6 + t4 * t9 - t5 * t9) + t3 * t6 * t7 + 2 * t7 * t8 * (ABx * beta * t6 - t5 * t6 - t9)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 237:
        t0 = p + q
        t1 = t0 ** 18.5 * F1
        t2 = CDz * gamma
        t3 = ABy * alpha
        t4 = t2 * t3
        t5 = p * q
        t6 = CDx * delta
        t7 = ABx * beta
        t8 = t6 * t7
        t9 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t10 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t11 = t10 * t9
        t12 = t0 ** 16.5 * F3
        t13 = t10 * t3
        t14 = t0 ** 17.5 * F2
        t15 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t16 = t15 ** 2
        t17 = t15 * t2 * t9
        return np.pi ** 2.5 * t0 ** (-20.0) * (2 * t0 ** 15.5 * t11 * t16 * F4 + 2 * t0 ** 19.5 * t4 * t8 * F0 + t1 * t4 * t5 + 2 * t1 * (ABx * ABy * CDz * alpha * beta * gamma * t15 + ABx * CDx * CDz * beta * delta * gamma * t9 - t13 * t8 - t15 * t4 * t6) - t11 * t12 * t5 + 2 * t12 * t15 * (ABy * alpha * t10 * t15 + CDx * delta * t10 * t9 - t11 * t7 - t17) + t14 * t5 * (CDz * gamma * t9 - t13) + 2 * t14 * (ABx * CDz * beta * gamma * t15 * t9 + ABy * CDx * alpha * delta * t10 * t15 - t11 * t6 * t7 - t13 * t15 * t7 - t16 * t4 - t17 * t6)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 238:
        t0 = p + q
        t1 = t0 ** 20.0 * F2
        t2 = ABx * beta
        t3 = CDx * delta
        t4 = t2 * t3
        t5 = CDy * gamma
        t6 = ABy * alpha
        t7 = t5 * t6
        t8 = t0 ** 21.0 * F1
        t9 = p * q
        t10 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t11 = t10 ** 2
        t12 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t13 = t12 ** 2
        t14 = t0 ** 19.0 * F3
        t15 = t10 * t2
        t16 = t12 * t5
        t17 = t12 * t6
        t18 = t10 * t3
        return np.pi ** 2.5 * t0 ** (-22.5) * (1 / 2 * p ** 2 * q ** 2 * t1 + 2 * t0 ** 18.0 * t11 * t13 * F4 + 2 * t0 ** 22.0 * t4 * t7 * F0 + t1 * t9 * (t15 + t16 - t17 - t18) + 2 * t1 * (ABx * CDy * beta * gamma * t10 * t12 + ABy * CDx * alpha * delta * t10 * t12 - t11 * t7 - t13 * t4 - t15 * t17 - t16 * t18) + 2 * t10 * t12 * t14 * (ABy * alpha * t10 + CDx * delta * t12 - t10 * t5 - t12 * t2) - t14 * t9 * (t11 + t13) + t8 * t9 * (t4 + t7) + 2 * t8 * (ABx * ABy * CDy * alpha * beta * gamma * t10 + ABx * CDx * CDy * beta * delta * gamma * t12 - t17 * t4 - t18 * t7)) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 239:
        t0 = p + q
        t1 = delta * gamma
        t2 = ABx * beta
        t3 = ABy * alpha
        t4 = t0 ** 24.0 * t2 * t3 * F0
        t5 = CDx ** 2 * t1
        t6 = CDx * gamma
        t7 = CDx * delta
        t8 = t0 ** 23.0 * F1
        t9 = q * t8
        t10 = p * (Cy * gamma + Dy * delta) - q * (Ay * alpha + By * beta)
        t11 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t12 = t0 ** 22.0 * F2
        t13 = t11 * t12
        t14 = t0 ** 21.0 * F3
        t15 = p * q
        t16 = t10 * t2
        t17 = -ABy * alpha * t11 + t16
        t18 = t11 * t3
        t19 = t10 * t6 - t10 * t7
        t20 = t18 * t2
        t21 = t18 * t6
        return np.pi ** 2.5 * t0 ** (-24.5) * (2 * CDx * t8 * (ABx * ABy * alpha * beta * gamma * t11 + ABx * CDx * beta * delta * gamma * t10 - delta * t20 - delta * t21) + p * t3 * t9 * (t2 + t6 - t7) + q * t10 * t13 - q * t4 + 2 * t0 ** 20.0 * t10 * t11 ** 3 * F4 - 3 * t10 * t11 * t14 * t15 + 2 * t11 ** 2 * t14 * (-t17 - t19) + t12 * t15 * (t16 - 3 * t18 + t19) + 2 * t13 * (ABx * CDx * beta * gamma * t10 + ABy * CDx * alpha * delta * t11 - t10 * t5 - t16 * t7 - t20 - t21) - t17 * t9 + 2 * t4 * t5) * KAB * KCD / (p ** 3 * q ** 3)
    if case_id == 240:
        t0 = q ** (-1.0)
        t1 = p + q
        t2 = alpha * beta
        t3 = F0
        t4 = t1 ** 6.0
        t5 = t1 ** 5.0 * F1
        t6 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        return np.pi ** 2.5 * t0 * t1 ** (-6.5) * (-2 * ABx ** 2 * t2 * t3 * t4 - 2 * ABx * t5 * t6 * (alpha - beta) - p * q * t5 + p * t3 * t4 + 2 * t1 ** 4.0 * t6 ** 2 * F2) * KAB * KCD / p ** 3
    if case_id == 241:
        t0 = p + q
        t1 = alpha * beta
        t2 = F0
        t3 = t0 ** 13.5
        t4 = t0 ** 12.5
        t5 = F1
        t6 = t4 * t5
        t7 = CDz * gamma
        t8 = -p * (Cz * gamma + Dz * delta) + q * (Az * alpha + Bz * beta)
        t9 = t0 ** 11.5 * F2
        t10 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t11 = ABx * t8
        t12 = t10 * t7
        return np.pi ** 2.5 * t0 ** (-14.0) * (-2 * ABx ** 2 * t1 * t2 * t3 * t7 - 2 * ABx * t6 * (alpha * beta * t11 + alpha * t12 - beta * t12) + CDz * gamma * p * t2 * t3 - p * q * t6 * t7 - p * q * t8 * t9 + p * t4 * t5 * t8 + 2 * t0 ** 10.5 * t10 ** 2 * t8 * F3 - 2 * t10 * t9 * (ABx * alpha * t8 - beta * t11 - t12)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 242:
        t0 = p + q
        t1 = alpha * beta
        t2 = F0
        t3 = t0 ** 13.5
        t4 = t0 ** 12.5
        t5 = F1
        t6 = t4 * t5
        t7 = CDy * gamma
        t8 = -p * (Cy * gamma + Dy * delta) + q * (Ay * alpha + By * beta)
        t9 = t0 ** 11.5 * F2
        t10 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t11 = ABx * t8
        t12 = t10 * t7
        return np.pi ** 2.5 * t0 ** (-14.0) * (-2 * ABx ** 2 * t1 * t2 * t3 * t7 - 2 * ABx * t6 * (alpha * beta * t11 + alpha * t12 - beta * t12) + CDy * gamma * p * t2 * t3 - p * q * t6 * t7 - p * q * t8 * t9 + p * t4 * t5 * t8 + 2 * t0 ** 10.5 * t10 ** 2 * t8 * F3 - 2 * t10 * t9 * (ABx * alpha * t8 - beta * t11 - t12)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 243:
        t0 = p + q
        t1 = F0
        t2 = t0 ** 13.5
        t3 = CDx * gamma
        t4 = alpha * t3
        t5 = -ABx * alpha + ABx * beta + t3
        t6 = t0 ** 12.5
        t7 = F1
        t8 = t6 * t7
        t9 = -p * (Cx * gamma + Dx * delta) + q * (Ax * alpha + Bx * beta)
        t10 = t0 ** 11.5 * F2
        return np.pi ** 2.5 * t0 ** (-14.0) * (-2 * ABx ** 2 * beta * t1 * t2 * t4 - 2 * ABx * t8 * t9 * (ABx * alpha * beta - beta * t3 + t4) + CDx * gamma * p * t1 * t2 - 3 * p * q * t10 * t9 - p * q * t5 * t8 + p * t6 * t7 * t9 + 2 * t0 ** 10.5 * t9 ** 3 * F3 + 2 * t10 * t5 * t9 ** 2) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 244:
        t0 = p + q
        t1 = alpha * beta
        t2 = CDz * delta
        t3 = p * t2
        t4 = t0 ** 13.5 * F0
        t5 = t0 ** 12.5 * F1
        t6 = p * (Cz * gamma + Dz * delta) - q * (Az * alpha + Bz * beta)
        t7 = p * t6
        t8 = t0 ** 11.5 * F2
        t9 = p * (Cx * gamma + Dx * delta) - q * (Ax * alpha + Bx * beta)
        t10 = ABx * t6
        t11 = t2 * t9
        return np.pi ** 2.5 * t0 ** (-14.0) * (2 * ABx ** 2 * t1 * t2 * t4 + 2 * ABx * t5 * (alpha * beta * t10 - alpha * t11 + beta * t11) + q * t3 * t5 + q * t7 * t8 - 2 * t0 ** 10.5 * t6 * t9 ** 2 * F3 - t3 * t4 - t5 * t7 + 2 * t8 * t9 * (ABx * beta * t6 - alpha * t10 - t11)) * KAB * KCD / (p ** 3 * q ** 2)
    if case_id == 245:
        t0 = p ** (-1.0)
        t1 = alpha * beta
        t2 = q ** (-1.0)
        t3 = delta * gamma
        t4 = p ** (-2.0)
        t5 = q ** (-2.0)
        t6 = t4 * t5
        t7 = p + q
        t8 = t7 ** (-0.5) * F0
        t9 = t7 ** (-2.5)
        t10 = F2
        t11 = t0 * t10 * t9
        t12 = t11 * t2
        t13 = t7 ** (-1.5) * F1
        t14 = t13 * t2 * t4
        t15 = t0 * t13 * t5
        t16 = 2 * t8
        t17 = ABx ** 2 * t1
        t18 = t17 / p ** 3
        t19 = t13 * t6
        t20 = 2 * t19
        t21 = CDz ** 2 * t3
        t22 = t21 / q ** 3
        t23 = t0 * (Ax * alpha + Bx * beta) + t2 * (-Cx * gamma - Dx * delta)
        t24 = t23 ** 2
        t25 = 2 * t24
        t26 = t7 ** (-3.5) * F3
        t27 = t0 * (Az * alpha + Bz * beta) + t2 * (-Cz * gamma - Dz * delta)
        t28 = t27 ** 2
        t29 = 2 * t28
        t30 = ABx * t23
        t31 = alpha * t30
        t32 = 2 * t31
        t33 = beta * t30
        t34 = 2 * t12
        t35 = CDz * t27
        t36 = delta * t35
        t37 = 2 * t15
        t38 = gamma * t35
        t39 = 4 * t28
        t40 = 4 * t12
        t41 = 4 * t19
        t42 = t21 * t41
        t43 = t17 * t41
        t44 = 4 * t26
        t45 = t28 * t44
        t46 = t24 * t44
        t47 = t31 * t40
        t48 = t33 * t40
        return 1 / 2 * np.pi ** 2.5 * (p * q * t24 * t39 * t7 ** (-4.5) * F4 + t10 * t2 * t29 * t9 + t11 * t25 - t12 * t17 * t39 + t12 * t32 + t12 - t14 * t32 + 2 * t14 * t33 - t14 - t15 - t16 * t18 * t5 - t16 * t22 * t4 + t17 * t20 + 4 * t18 * t22 * t8 + t20 * t21 - t21 * t24 * t40 - t25 * t26 - t26 * t29 + t31 * t42 - t31 * t45 - t33 * t34 - t33 * t42 + t33 * t45 + t34 * t36 - t34 * t38 - t36 * t37 + t36 * t43 - t36 * t46 + t36 * t47 - t36 * t48 + t37 * t38 - t38 * t43 + t38 * t46 - t38 * t47 + t38 * t48 + t6 * t8) * KAB * KCD
    if case_id == 246:
        t0 = p ** (-1.0)
        t1 = alpha * t0
        t2 = beta * t1
        t3 = q ** (-1.0)
        t4 = delta * t3
        t5 = gamma * t4
        t6 = p ** (-2.0)
        t7 = p + q
        t8 = t7 ** (-1.5) * F1 / q ** 2
        t9 = t6 * t8
        t10 = CDy * gamma
        t11 = CDz * delta * t10
        t12 = t11 * t9
        t13 = t11 * t7 ** (-0.5) * F0 / q ** 3
        t14 = ABx ** 2
        t15 = 2 * alpha
        t16 = beta * t14 * t15
        t17 = t0 * (Az * alpha + Bz * beta) + t3 * (-Cz * gamma - Dz * delta)
        t18 = t10 * t17
        t19 = t0 * t18
        t20 = t7 ** (-2.5) * F2
        t21 = t0 * t20
        t22 = t0 * (Ay * alpha + By * beta) + t3 * (-Cy * gamma - Dy * delta)
        t23 = CDz * t22
        t24 = t23 * t4
        t25 = t21 * t24
        t26 = t20 * t3
        t27 = t19 * t26
        t28 = delta * t23
        t29 = t28 * t8
        t30 = t0 * (Ax * alpha + Bx * beta) + t3 * (-Cx * gamma - Dx * delta)
        t31 = 2 * t30 ** 2
        t32 = ABx * t30
        t33 = 2 * beta
        t34 = t32 * t33
        t35 = t17 * t22
        t36 = t7 ** (-3.5) * F3
        t37 = t35 * t36
        t38 = t26 * t35
        t39 = t31 * t36
        t40 = 2 * t1 * t32
        t41 = t32 * t37
        return np.pi ** 2.5 * (-CDy * CDz * t21 * t31 * t5 + p * q * t31 * t35 * t7 ** (-4.5) * F4 - t0 * t29 + t12 * t15 * t32 - t12 * t34 + t12 - t13 * t6 - 2 * t14 * t2 * t38 - t15 * t41 - t16 * t18 * t9 + t16 * t29 * t6 - t18 * t26 * t40 + t18 * t39 + t19 * t8 + t20 * t24 * t40 - t25 * t34 + t25 + t27 * t34 - t27 - t28 * t39 + t33 * t41 - t37 + t38 + t13 * t16 / p ** 3) * KAB * KCD
    if case_id == 247:
        t0 = p ** (-1.0)
        t1 = alpha * t0
        t2 = beta * t1
        t3 = q ** (-1.0)
        t4 = gamma * t3
        t5 = delta * t4
        t6 = p + q
        t7 = t6 ** (-1.5) * F1 / q ** 2
        t8 = p ** (-2.0)
        t9 = CDz * delta
        t10 = t8 * t9
        t11 = t10 * t7
        t12 = ABx * t11
        t13 = CDx * gamma
        t14 = t13 * t6 ** (-0.5) * F0 / q ** 3
        t15 = 2 * ABx ** 2
        t16 = alpha * t15
        t17 = t0 * (Ax * alpha + Bx * beta) + t3 * (-Cx * gamma - Dx * delta)
        t18 = t17 ** 3
        t19 = t6 ** (-3.5) * F3
        t20 = 2 * t19
        t21 = t6 ** (-2.5) * F2
        t22 = t21 * t3
        t23 = t0 * (Az * alpha + Bz * beta) + t3 * (-Cz * gamma - Dz * delta)
        t24 = ABx * t23
        t25 = t22 * t24
        t26 = CDx * t23
        t27 = gamma * t26
        t28 = t0 * t7
        t29 = beta * t0
        t30 = t0 * t21
        t31 = t17 * t9
        t32 = t17 ** 2
        t33 = 2 * t32
        t34 = t22 * t33 * t9
        t35 = ABx * t1
        t36 = ABx * t29
        t37 = 2 * ABx * t13 * t31
        t38 = beta * t7 * t8
        t39 = t16 * t38
        t40 = t17 * t23
        t41 = t22 * t40
        t42 = t20 * t32
        t43 = t24 * t42
        t44 = 2 * CDx * t21 * t4 * t40
        return np.pi ** 2.5 * (-CDx * CDz * t30 * t33 * t5 - alpha * t12 + alpha * t37 * t7 * t8 - alpha * t43 + beta * t12 + beta * t43 + beta * t14 * t16 * t9 / p ** 3 + 2 * p * q * t18 * t23 * t6 ** (-4.5) * F4 + 3 * t0 * t22 * t31 + t1 * t25 - t10 * t14 + t11 * t13 - t15 * t2 * t41 - t18 * t20 * t9 - 3 * t19 * t40 - t25 * t29 - t26 * t30 * t4 + t27 * t28 - t27 * t39 + t27 * t42 - t28 * t31 + t31 * t39 + t34 * t35 - t34 * t36 - t35 * t44 + t36 * t44 - t37 * t38 + t41) * KAB * KCD
    if case_id == 248:
        t0 = p ** (-1.0)
        t1 = alpha * beta
        t2 = q ** (-1.0)
        t3 = delta * t2
        t4 = p + q
        t5 = t4 ** (-1.5) * F1
        t6 = p ** (-2.0)
        t7 = CDy * t6
        t8 = t3 * t5 * t7
        t9 = delta * t4 ** (-0.5) * F0 / q ** 2
        t10 = 2 * ABx ** 2 * t1
        t11 = t0 * (Ay * alpha + By * beta) + t2 * (-Cy * gamma - Dy * delta)
        t12 = t0 * t11
        t13 = t4 ** (-2.5) * F2
        t14 = t12 * t13
        t15 = t2 * t5
        t16 = t0 * (Ax * alpha + Bx * beta) + t2 * (-Cx * gamma - Dx * delta)
        t17 = 2 * t16 ** 2
        t18 = 2 * ABx * t16
        t19 = t18 * t8
        t20 = t14 * t18
        return np.pi ** 2.5 * (-CDy * delta * t0 * t13 * t17 + CDy * t10 * t9 / p ** 3 + alpha * t19 - alpha * t20 - beta * t19 + beta * t20 + q * t11 * t17 * t4 ** (-3.5) * F3 - t10 * t11 * t15 * t6 + t12 * t15 - t14 - t7 * t9 + t8) * KAB * KCD
    if case_id == 249:
        t0 = p ** (-1.0)
        t1 = beta * t0
        t2 = alpha * t1
        t3 = q ** (-1.0)
        t4 = delta * t3
        t5 = gamma * t4
        t6 = p ** (-2.0)
        t7 = p + q
        t8 = t7 ** (-1.5) * F1 / q ** 2
        t9 = t6 * t8
        t10 = CDz * gamma
        t11 = CDy * delta * t10
        t12 = t11 * t9
        t13 = t11 * t7 ** (-0.5) * F0 / q ** 3
        t14 = ABx ** 2
        t15 = 2 * alpha
        t16 = beta * t14 * t15
        t17 = t7 ** (-2.5) * F2
        t18 = t0 * t17
        t19 = t0 * (Az * alpha + Bz * beta) + t3 * (-Cz * gamma - Dz * delta)
        t20 = CDy * t19
        t21 = t20 * t4
        t22 = t18 * t21
        t23 = t0 * (Ay * alpha + By * beta) + t3 * (-Cy * gamma - Dy * delta)
        t24 = t10 * t23
        t25 = t0 * t24
        t26 = delta * t20
        t27 = t26 * t8
        t28 = t17 * t3
        t29 = t25 * t28
        t30 = t0 * (Ax * alpha + Bx * beta) + t3 * (-Cx * gamma - Dx * delta)
        t31 = 2 * t30 ** 2
        t32 = ABx * t30
        t33 = t15 * t32
        t34 = 2 * beta
        t35 = t19 * t23
        t36 = t7 ** (-3.5) * F3
        t37 = t35 * t36
        t38 = t28 * t35
        t39 = t31 * t36
        t40 = 2 * t1 * t32
        t41 = t32 * t37
        return np.pi ** 2.5 * (-CDy * CDz * t18 * t31 * t5 + p * q * t31 * t35 * t7 ** (-4.5) * F4 - t0 * t27 - t12 * t32 * t34 + t12 * t33 + t12 - t13 * t6 - 2 * t14 * t2 * t38 - t15 * t41 - t16 * t24 * t9 + t16 * t27 * t6 - t17 * t21 * t40 + t22 * t33 + t22 + t24 * t28 * t40 + t24 * t39 + t25 * t8 - t26 * t39 - t29 * t33 - t29 + t34 * t41 - t37 + t38 + t13 * t16 / p ** 3) * KAB * KCD
    if case_id == 250:
        t0 = p ** (-1.0)
        t1 = alpha * beta
        t2 = q ** (-1.0)
        t3 = delta * gamma
        t4 = p ** (-2.0)
        t5 = q ** (-2.0)
        t6 = t4 * t5
        t7 = p + q
        t8 = t7 ** (-0.5) * F0
        t9 = t7 ** (-2.5)
        t10 = F2
        t11 = t0 * t10 * t9
        t12 = t11 * t2
        t13 = t7 ** (-1.5) * F1
        t14 = t13 * t2 * t4
        t15 = t0 * t13 * t5
        t16 = 2 * t8
        t17 = ABx ** 2 * t1
        t18 = t17 / p ** 3
        t19 = t13 * t6
        t20 = 2 * t19
        t21 = CDy ** 2 * t3
        t22 = t21 / q ** 3
        t23 = t0 * (Ax * alpha + Bx * beta) + t2 * (-Cx * gamma - Dx * delta)
        t24 = t23 ** 2
        t25 = 2 * t24
        t26 = t7 ** (-3.5) * F3
        t27 = t0 * (Ay * alpha + By * beta) + t2 * (-Cy * gamma - Dy * delta)
        t28 = t27 ** 2
        t29 = 2 * t28
        t30 = ABx * t23
        t31 = alpha * t30
        t32 = 2 * t31
        t33 = beta * t30
        t34 = 2 * t12
        t35 = CDy * t27
        t36 = delta * t35
        t37 = 2 * t15
        t38 = gamma * t35
        t39 = 4 * t28
        t40 = 4 * t12
        t41 = 4 * t19
        t42 = t21 * t41
        t43 = t17 * t41
        t44 = 4 * t26
        t45 = t28 * t44
        t46 = t24 * t44
        t47 = t31 * t40
        t48 = t33 * t40
        return 1 / 2 * np.pi ** 2.5 * (p * q * t24 * t39 * t7 ** (-4.5) * F4 + t10 * t2 * t29 * t9 + t11 * t25 - t12 * t17 * t39 + t12 * t32 + t12 - t14 * t32 + 2 * t14 * t33 - t14 - t15 - t16 * t18 * t5 - t16 * t22 * t4 + t17 * t20 + 4 * t18 * t22 * t8 + t20 * t21 - t21 * t24 * t40 - t25 * t26 - t26 * t29 + t31 * t42 - t31 * t45 - t33 * t34 - t33 * t42 + t33 * t45 + t34 * t36 - t34 * t38 - t36 * t37 + t36 * t43 - t36 * t46 + t36 * t47 - t36 * t48 + t37 * t38 - t38 * t43 + t38 * t46 - t38 * t47 + t38 * t48 + t6 * t8) * KAB * KCD
    if case_id == 251:
        t0 = p ** (-1.0)
        t1 = alpha * t0
        t2 = beta * t1
        t3 = q ** (-1.0)
        t4 = gamma * t3
        t5 = delta * t4
        t6 = p + q
        t7 = t6 ** (-1.5) * F1 / q ** 2
        t8 = p ** (-2.0)
        t9 = CDy * delta
        t10 = t8 * t9
        t11 = t10 * t7
        t12 = ABx * t11
        t13 = CDx * gamma
        t14 = t13 * t6 ** (-0.5) * F0 / q ** 3
        t15 = 2 * ABx ** 2
        t16 = alpha * t15
        t17 = t0 * (Ax * alpha + Bx * beta) + t3 * (-Cx * gamma - Dx * delta)
        t18 = t17 ** 3
        t19 = t6 ** (-3.5) * F3
        t20 = 2 * t19
        t21 = t6 ** (-2.5) * F2
        t22 = t21 * t3
        t23 = t0 * (Ay * alpha + By * beta) + t3 * (-Cy * gamma - Dy * delta)
        t24 = ABx * t23
        t25 = t22 * t24
        t26 = CDx * t23
        t27 = gamma * t26
        t28 = t0 * t7
        t29 = beta * t0
        t30 = t0 * t21
        t31 = t17 * t9
        t32 = t17 ** 2
        t33 = 2 * t32
        t34 = t22 * t33 * t9
        t35 = ABx * t1
        t36 = ABx * t29
        t37 = 2 * ABx * t13 * t31
        t38 = beta * t7 * t8
        t39 = t16 * t38
        t40 = t17 * t23
        t41 = t22 * t40
        t42 = t20 * t32
        t43 = t24 * t42
        t44 = 2 * CDx * t21 * t4 * t40
        return np.pi ** 2.5 * (-CDx * CDy * t30 * t33 * t5 - alpha * t12 + alpha * t37 * t7 * t8 - alpha * t43 + beta * t12 + beta * t43 + beta * t14 * t16 * t9 / p ** 3 + 2 * p * q * t18 * t23 * t6 ** (-4.5) * F4 + 3 * t0 * t22 * t31 + t1 * t25 - t10 * t14 + t11 * t13 - t15 * t2 * t41 - t18 * t20 * t9 - 3 * t19 * t40 - t25 * t29 - t26 * t30 * t4 + t27 * t28 - t27 * t39 + t27 * t42 - t28 * t31 + t31 * t39 + t34 * t35 - t34 * t36 - t35 * t44 + t36 * t44 - t37 * t38 + t41) * KAB * KCD
    if case_id == 252:
        t0 = p ** (-1.0)
        t1 = alpha * t0
        t2 = q ** (-1.0)
        t3 = delta * t2
        t4 = p + q
        t5 = t4 ** (-1.5) * F1
        t6 = t2 * t5
        t7 = p ** (-2.0)
        t8 = ABx * t7
        t9 = t6 * t8
        t10 = CDx * t7
        t11 = t3 * t5
        t12 = delta * t4 ** (-0.5) * F0 / q ** 2
        t13 = ABx ** 2 * alpha
        t14 = t0 * (Ax * alpha + Bx * beta) + t2 * (-Cx * gamma - Dx * delta)
        t15 = t0 * t14
        t16 = t4 ** (-2.5) * F2
        t17 = 2 * t14 ** 2 * t16
        t18 = ABx * t17
        t19 = CDx * t11 * t8
        t20 = 2 * beta * t14
        return np.pi ** 2.5 * (2 * CDx * beta * t12 * t13 / p ** 3 - CDx * delta * t0 * t17 + 2 * alpha * t14 * t19 + alpha * t9 + beta * t0 * t18 - beta * t9 + 2 * q * t14 ** 3 * t4 ** (-3.5) * F3 - t1 * t18 + t10 * t11 - t10 * t12 - t13 * t20 * t6 * t7 - 3 * t15 * t16 + t15 * t6 - t19 * t20) * KAB * KCD
    if case_id == 253:
        t0 = p ** (-1.0)
        t1 = alpha * t0
        t2 = beta * t1
        t3 = q ** (-1.0)
        t4 = delta * t3
        t5 = gamma * t4
        t6 = p + q
        t7 = t6 ** (-1.5) * F1 / q ** 2
        t8 = p ** (-2.0)
        t9 = CDz * gamma
        t10 = t8 * t9
        t11 = t10 * t7
        t12 = ABx * t11
        t13 = alpha * t12
        t14 = CDx * delta
        t15 = beta * t12
        t16 = t14 * t6 ** (-0.5) * F0 / q ** 3
        t17 = 2 * ABx ** 2
        t18 = alpha * beta * t17
        t19 = t0 * (Ax * alpha + Bx * beta) + t3 * (-Cx * gamma - Dx * delta)
        t20 = t19 ** 3
        t21 = t6 ** (-3.5) * F3
        t22 = 2 * t21
        t23 = t6 ** (-2.5) * F2
        t24 = t23 * t3
        t25 = t0 * (Az * alpha + Bz * beta) + t3 * (-Cz * gamma - Dz * delta)
        t26 = ABx * t25
        t27 = t24 * t26
        t28 = CDx * t25
        t29 = t0 * t23
        t30 = t29 * t4
        t31 = t0 * t9
        t32 = t19 * t31
        t33 = delta * t28
        t34 = t33 * t7
        t35 = ABx * t1
        t36 = t19 ** 2
        t37 = 2 * t36
        t38 = t24 * t37
        t39 = ABx * beta
        t40 = 2 * t14 * t19
        t41 = t19 * t25
        t42 = t24 * t41
        t43 = t22 * t36
        t44 = t26 * t43
        t45 = 2 * CDx * t41
        return np.pi ** 2.5 * (-CDx * CDz * t29 * t37 * t5 - alpha * t44 - beta * t0 * t27 + beta * t44 + 2 * p * q * t20 * t25 * t6 ** (-4.5) * F4 - t0 * t34 + t1 * t27 - t10 * t16 + t11 * t14 - t11 * t18 * t19 + t13 * t40 + t13 - t15 * t40 - t15 - t17 * t2 * t42 + t18 * t34 * t8 + t20 * t22 * t9 - 3 * t21 * t41 + t23 * t35 * t4 * t45 - 3 * t24 * t32 + t28 * t30 - t30 * t39 * t45 + t31 * t38 * t39 + t32 * t7 - t33 * t43 - t35 * t38 * t9 + t42 + t16 * t18 * t9 / p ** 3) * KAB * KCD
    if case_id == 254:
        t0 = p ** (-1.0)
        t1 = alpha * t0
        t2 = beta * t1
        t3 = q ** (-1.0)
        t4 = delta * t3
        t5 = gamma * t4
        t6 = p + q
        t7 = t6 ** (-1.5) * F1 / q ** 2
        t8 = p ** (-2.0)
        t9 = CDy * gamma
        t10 = t8 * t9
        t11 = t10 * t7
        t12 = ABx * t11
        t13 = alpha * t12
        t14 = CDx * delta
        t15 = beta * t12
        t16 = t14 * t6 ** (-0.5) * F0 / q ** 3
        t17 = 2 * ABx ** 2
        t18 = alpha * beta * t17
        t19 = t0 * (Ax * alpha + Bx * beta) + t3 * (-Cx * gamma - Dx * delta)
        t20 = t19 ** 3
        t21 = t6 ** (-3.5) * F3
        t22 = 2 * t21
        t23 = t6 ** (-2.5) * F2
        t24 = t23 * t3
        t25 = t0 * (Ay * alpha + By * beta) + t3 * (-Cy * gamma - Dy * delta)
        t26 = ABx * t25
        t27 = t24 * t26
        t28 = CDx * t25
        t29 = t0 * t23
        t30 = t29 * t4
        t31 = t0 * t9
        t32 = t19 * t31
        t33 = delta * t28
        t34 = t33 * t7
        t35 = ABx * t1
        t36 = t19 ** 2
        t37 = 2 * t36
        t38 = t24 * t37
        t39 = ABx * beta
        t40 = 2 * t14 * t19
        t41 = t19 * t25
        t42 = t24 * t41
        t43 = t22 * t36
        t44 = t26 * t43
        t45 = 2 * CDx * t41
        return np.pi ** 2.5 * (-CDx * CDy * t29 * t37 * t5 - alpha * t44 - beta * t0 * t27 + beta * t44 + 2 * p * q * t20 * t25 * t6 ** (-4.5) * F4 - t0 * t34 + t1 * t27 - t10 * t16 + t11 * t14 - t11 * t18 * t19 + t13 * t40 + t13 - t15 * t40 - t15 - t17 * t2 * t42 + t18 * t34 * t8 + t20 * t22 * t9 - 3 * t21 * t41 + t23 * t35 * t4 * t45 - 3 * t24 * t32 + t28 * t30 - t30 * t39 * t45 + t31 * t38 * t39 + t32 * t7 - t33 * t43 - t35 * t38 * t9 + t42 + t16 * t18 * t9 / p ** 3) * KAB * KCD
    if case_id == 255:
        t0 = p ** (-1.0)
        t1 = beta * t0
        t2 = alpha * t1
        t3 = q ** (-1.0)
        t4 = gamma * t3
        t5 = delta * t4
        t6 = p ** (-2.0)
        t7 = q ** (-2.0)
        t8 = t6 * t7
        t9 = p + q
        t10 = t9 ** (-0.5) * F0
        t11 = t9 ** (-1.5) * F1
        t12 = t11 * t3 * t6
        t13 = t0 * t11 * t7
        t14 = t9 ** (-2.5) * F2
        t15 = t0 * t14
        t16 = t15 * t3
        t17 = CDx * delta
        t18 = ABx * alpha
        t19 = t11 * t8
        t20 = 2 * t19
        t21 = t18 * t20
        t22 = CDx * gamma
        t23 = ABx * beta
        t24 = t20 * t23
        t25 = 2 * t10
        t26 = ABx ** 2
        t27 = alpha * beta * t26
        t28 = t27 / p ** 3
        t29 = CDx ** 2
        t30 = delta * gamma * t29
        t31 = t30 / q ** 3
        t32 = t0 * (Ax * alpha + Bx * beta) + t3 * (-Cx * gamma - Dx * delta)
        t33 = t32 ** 2
        t34 = t9 ** (-3.5) * F3
        t35 = 4 * t32 ** 3 * t34
        t36 = CDx * t35
        t37 = 2 * t33
        t38 = t14 * t3
        t39 = 2 * t32
        t40 = t12 * t39
        t41 = 6 * t32
        t42 = t16 * t18
        t43 = ABx * t1
        t44 = t38 * t43
        t45 = t13 * t39
        t46 = CDx * t41
        t47 = t15 * t4
        t48 = 4 * t33
        t49 = t17 * t48
        t50 = CDx * t48
        t51 = 4 * t19 * t32
        t52 = t30 * t51
        t53 = t27 * t51
        return 1 / 2 * np.pi ** 2.5 * (delta * t16 * t46 - delta * t36 + gamma * t36 + 4 * p * q * t32 ** 4 * t9 ** (-4.5) * F4 + 4 * t10 * t28 * t31 + t10 * t8 - t12 - t13 + t14 * t4 * t43 * t50 - t15 * t29 * t48 * t5 + t15 * t37 + 3 * t16 - t17 * t21 + t17 * t24 - t17 * t45 + t17 * t53 - t18 * t35 - t18 * t40 - t18 * t47 * t50 + t18 * t52 - t2 * t26 * t38 * t48 + t20 * t27 + t20 * t30 + t21 * t22 - t22 * t24 + t22 * t45 - t22 * t53 + t23 * t35 + t23 * t40 - t23 * t52 - t25 * t28 * t7 - t25 * t31 * t6 - 12 * t33 * t34 + t37 * t38 + t41 * t42 - t41 * t44 + t42 * t49 - t44 * t49 - t46 * t47) * KAB * KCD
    raise KeyError(case_id)
