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
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        return 2*np.pi**2.5*F0*t0*t1*(p + q)**(-0.5)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 1:
        t0 = q**(-1.0)
        t1 = p**(-1.0)
        t2 = gamma*t0
        t3 = p + q
        return 2*np.pi**2.5*t0*(CDz*F0*t1*t2*t3**(-0.5) + F1*t3**(-1.5)*(-t0*(Cz*gamma + Dz*delta) + t1*(Az*alpha + Bz*beta)))*np.exp(-alpha*beta*rAB*t1 - delta*rCD*t2)
    if case_id == 2:
        t0 = q**(-1.0)
        t1 = p**(-1.0)
        t2 = gamma*t0
        t3 = p + q
        return 2*np.pi**2.5*t0*(CDy*F0*t1*t2*t3**(-0.5) + F1*t3**(-1.5)*(-t0*(Cy*gamma + Dy*delta) + t1*(Ay*alpha + By*beta)))*np.exp(-alpha*beta*rAB*t1 - delta*rCD*t2)
    if case_id == 3:
        t0 = q**(-1.0)
        t1 = p**(-1.0)
        t2 = gamma*t0
        t3 = p + q
        return 2*np.pi**2.5*t0*(CDx*F0*t1*t2*t3**(-0.5) + F1*t3**(-1.5)*(-t0*(Cx*gamma + Dx*delta) + t1*(Ax*alpha + Bx*beta)))*np.exp(-alpha*beta*rAB*t1 - delta*rCD*t2)
    if case_id == 4:
        t0 = q**(-1.0)
        t1 = p**(-1.0)
        t2 = delta*t0
        t3 = p + q
        return 2*np.pi**2.5*t0*(-CDz*F0*t1*t2*t3**(-0.5) + F1*t3**(-1.5)*(-t0*(Cz*gamma + Dz*delta) + t1*(Az*alpha + Bz*beta)))*np.exp(-alpha*beta*rAB*t1 - gamma*rCD*t2)
    if case_id == 5:
        t0 = q**(-1.0)
        t1 = p**(-1.0)
        t2 = delta*gamma
        t3 = p + q
        t4 = t3**(-1.5)
        t5 = F1*t0*t4
        t6 = t3**(-0.5)
        t7 = -t0*(Cz*gamma + Dz*delta) + t1*(Az*alpha + Bz*beta)
        return np.pi**2.5*t0*(-2*CDz**2*F0*t1*t2*t6/q**2 + 2*CDz*F1*gamma*t0*t4*t7 - 2*CDz*delta*t5*t7 + F0*t0*t1*t6 + 2*F2*p*t3**(-2.5)*t7**2 - t5)*np.exp(-alpha*beta*rAB*t1 - rCD*t0*t2)
    if case_id == 6:
        t0 = q**(-1.0)
        t1 = p**(-1.0)
        t2 = p + q
        t3 = CDz*delta
        t4 = -t0*(Cy*gamma + Dy*delta) + t1*(Ay*alpha + By*beta)
        t5 = t2**(-1.5)
        t6 = -t0*(Cz*gamma + Dz*delta) + t1*(Az*alpha + Bz*beta)
        return 2*np.pi**2.5*t0*(-CDy*F0*gamma*t1*t2**(-0.5)*t3/q**2 + CDy*F1*gamma*t0*t5*t6 - F1*t0*t3*t4*t5 + F2*p*t2**(-2.5)*t4*t6)*np.exp(-alpha*beta*rAB*t1 - delta*gamma*rCD*t0)
    if case_id == 7:
        t0 = q**(-1.0)
        t1 = p**(-1.0)
        t2 = p + q
        t3 = CDz*delta
        t4 = -t0*(Cx*gamma + Dx*delta) + t1*(Ax*alpha + Bx*beta)
        t5 = t2**(-1.5)
        t6 = -t0*(Cz*gamma + Dz*delta) + t1*(Az*alpha + Bz*beta)
        return 2*np.pi**2.5*t0*(-CDx*F0*gamma*t1*t2**(-0.5)*t3/q**2 + CDx*F1*gamma*t0*t5*t6 - F1*t0*t3*t4*t5 + F2*p*t2**(-2.5)*t4*t6)*np.exp(-alpha*beta*rAB*t1 - delta*gamma*rCD*t0)
    if case_id == 8:
        t0 = q**(-1.0)
        t1 = p**(-1.0)
        t2 = delta*t0
        t3 = p + q
        return 2*np.pi**2.5*t0*(-CDy*F0*t1*t2*t3**(-0.5) + F1*t3**(-1.5)*(-t0*(Cy*gamma + Dy*delta) + t1*(Ay*alpha + By*beta)))*np.exp(-alpha*beta*rAB*t1 - gamma*rCD*t2)
    if case_id == 9:
        t0 = q**(-1.0)
        t1 = p**(-1.0)
        t2 = p + q
        t3 = CDy*delta
        t4 = -t0*(Cz*gamma + Dz*delta) + t1*(Az*alpha + Bz*beta)
        t5 = t2**(-1.5)
        t6 = -t0*(Cy*gamma + Dy*delta) + t1*(Ay*alpha + By*beta)
        return 2*np.pi**2.5*t0*(-CDz*F0*gamma*t1*t2**(-0.5)*t3/q**2 + CDz*F1*gamma*t0*t5*t6 - F1*t0*t3*t4*t5 + F2*p*t2**(-2.5)*t4*t6)*np.exp(-alpha*beta*rAB*t1 - delta*gamma*rCD*t0)
    if case_id == 10:
        t0 = q**(-1.0)
        t1 = p**(-1.0)
        t2 = delta*gamma
        t3 = p + q
        t4 = t3**(-1.5)
        t5 = F1*t0*t4
        t6 = t3**(-0.5)
        t7 = -t0*(Cy*gamma + Dy*delta) + t1*(Ay*alpha + By*beta)
        return np.pi**2.5*t0*(-2*CDy**2*F0*t1*t2*t6/q**2 + 2*CDy*F1*gamma*t0*t4*t7 - 2*CDy*delta*t5*t7 + F0*t0*t1*t6 + 2*F2*p*t3**(-2.5)*t7**2 - t5)*np.exp(-alpha*beta*rAB*t1 - rCD*t0*t2)
    if case_id == 11:
        t0 = q**(-1.0)
        t1 = p**(-1.0)
        t2 = p + q
        t3 = CDy*delta
        t4 = -t0*(Cx*gamma + Dx*delta) + t1*(Ax*alpha + Bx*beta)
        t5 = t2**(-1.5)
        t6 = -t0*(Cy*gamma + Dy*delta) + t1*(Ay*alpha + By*beta)
        return 2*np.pi**2.5*t0*(-CDx*F0*gamma*t1*t2**(-0.5)*t3/q**2 + CDx*F1*gamma*t0*t5*t6 - F1*t0*t3*t4*t5 + F2*p*t2**(-2.5)*t4*t6)*np.exp(-alpha*beta*rAB*t1 - delta*gamma*rCD*t0)
    if case_id == 12:
        t0 = q**(-1.0)
        t1 = p**(-1.0)
        t2 = delta*t0
        t3 = p + q
        return 2*np.pi**2.5*t0*(-CDx*F0*t1*t2*t3**(-0.5) + F1*t3**(-1.5)*(-t0*(Cx*gamma + Dx*delta) + t1*(Ax*alpha + Bx*beta)))*np.exp(-alpha*beta*rAB*t1 - gamma*rCD*t2)
    if case_id == 13:
        t0 = q**(-1.0)
        t1 = p**(-1.0)
        t2 = p + q
        t3 = CDx*delta
        t4 = -t0*(Cz*gamma + Dz*delta) + t1*(Az*alpha + Bz*beta)
        t5 = t2**(-1.5)
        t6 = -t0*(Cx*gamma + Dx*delta) + t1*(Ax*alpha + Bx*beta)
        return 2*np.pi**2.5*t0*(-CDz*F0*gamma*t1*t2**(-0.5)*t3/q**2 + CDz*F1*gamma*t0*t5*t6 - F1*t0*t3*t4*t5 + F2*p*t2**(-2.5)*t4*t6)*np.exp(-alpha*beta*rAB*t1 - delta*gamma*rCD*t0)
    if case_id == 14:
        t0 = q**(-1.0)
        t1 = p**(-1.0)
        t2 = p + q
        t3 = CDx*delta
        t4 = -t0*(Cy*gamma + Dy*delta) + t1*(Ay*alpha + By*beta)
        t5 = t2**(-1.5)
        t6 = -t0*(Cx*gamma + Dx*delta) + t1*(Ax*alpha + Bx*beta)
        return 2*np.pi**2.5*t0*(-CDy*F0*gamma*t1*t2**(-0.5)*t3/q**2 + CDy*F1*gamma*t0*t5*t6 - F1*t0*t3*t4*t5 + F2*p*t2**(-2.5)*t4*t6)*np.exp(-alpha*beta*rAB*t1 - delta*gamma*rCD*t0)
    if case_id == 15:
        t0 = q**(-1.0)
        t1 = p**(-1.0)
        t2 = delta*gamma
        t3 = p + q
        t4 = t3**(-1.5)
        t5 = F1*t0*t4
        t6 = t3**(-0.5)
        t7 = -t0*(Cx*gamma + Dx*delta) + t1*(Ax*alpha + Bx*beta)
        return np.pi**2.5*t0*(-2*CDx**2*F0*t1*t2*t6/q**2 + 2*CDx*F1*gamma*t0*t4*t7 - 2*CDx*delta*t5*t7 + F0*t0*t1*t6 + 2*F2*p*t3**(-2.5)*t7**2 - t5)*np.exp(-alpha*beta*rAB*t1 - rCD*t0*t2)
    if case_id == 16:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = p + q
        return 2*np.pi**2.5*t0*(ABz*F0*t1*t2*t3**(-0.5) - F1*t3**(-1.5)*(t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)))*np.exp(-beta*rAB*t1 - delta*gamma*rCD*t2)
    if case_id == 17:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = F1*t0*t1*t2**(-1.5)
        t4 = CDz*gamma
        t5 = ABz*alpha
        t6 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        t7 = 2*t3*t6
        return np.pi**2.5*(2*F0*t2**(-0.5)*t4*t5/(p**2*q**2) - 2*F2*t2**(-2.5)*t6**2 + t3 - t4*t7 + t5*t7)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 18:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        return 2*np.pi**2.5*(ABz*CDy*F0*alpha*gamma*t4**(-0.5)/(p**2*q**2) + ABz*t1*t2*t5*t6 - CDy*t0*t3*t6*t7 - F2*t4**(-2.5)*t5*t7)*np.exp(-beta*rAB*t1 - delta*rCD*t3)
    if case_id == 19:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        return 2*np.pi**2.5*(ABz*CDx*F0*alpha*gamma*t4**(-0.5)/(p**2*q**2) + ABz*t1*t2*t5*t6 - CDx*t0*t3*t6*t7 - F2*t4**(-2.5)*t5*t7)*np.exp(-beta*rAB*t1 - delta*rCD*t3)
    if case_id == 20:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = F1*t0*t1*t2**(-1.5)
        t4 = ABz*alpha
        t5 = CDz*delta
        t6 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        t7 = 2*t3*t6
        return np.pi**2.5*(-2*F0*t2**(-0.5)*t4*t5/(p**2*q**2) - 2*F2*t2**(-2.5)*t6**2 + t3 + t4*t7 + t5*t7)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 21:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = q**(-2.0)
        t8 = ABz*t1*t6*t7
        t9 = t0*t6
        t10 = CDz*delta
        t11 = p**(-2.0)
        t12 = t4**(-0.5)
        t13 = CDz**2
        t14 = t4**(-2.5)
        t15 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t16 = t15**2
        return np.pi**2.5*(2*ABz*CDz*F1*alpha*gamma*t0*t15*t5*t7 - 2*ABz*F0*alpha*delta*gamma*t11*t12*t13/q**3 + ABz*F0*alpha*t11*t12*t7 + 2*ABz*F2*alpha*t14*t16*t2 + CDz*F1*gamma*t0*t5*t7 + 2*CDz*F2*delta*t14*t16*t2 - 2*CDz*F2*t14*t16*t3 + 2*F1*delta*gamma*t0*t13*t15*t5*t7 + 3*F2*t14*t15*t2 - 2*F3*p*t15**3*t4**(-3.5) - 2*t10*t15*t8 - t10*t7*t9 - t15*t2*t9 - t8)*np.exp(-beta*rAB*t1 - delta*rCD*t3)
    if case_id == 22:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDy*gamma
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABz
        t9 = alpha*t8
        t10 = CDy*CDz*t3
        t11 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t15 = 2*t14**2
        t16 = t1*t8
        t17 = CDz*delta
        t18 = t13*t14
        return np.pi**2.5*(-F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 + t0*t7 - t11*t16*t17*t6 - t12*t15*t4 + t13 + t14*t16*t7 + 2*t17*t18 + t18*t9)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 23:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDx*gamma
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABz
        t9 = alpha*t8
        t10 = CDx*CDz*t3
        t11 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t15 = 2*t14**2
        t16 = t1*t8
        t17 = CDz*delta
        t18 = t13*t14
        return np.pi**2.5*(-F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 + t0*t7 - t11*t16*t17*t6 - t12*t15*t4 + t13 + t14*t16*t7 + 2*t17*t18 + t18*t9)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 24:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = t2**(-1.5)
        t4 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        t5 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        return 2*np.pi**2.5*(-ABz*CDy*F0*alpha*delta*t2**(-0.5)/(p**2*q**2) + ABz*F1*alpha*t0*t1*t3*t4 + CDy*F1*delta*t0*t1*t3*t5 - F2*t2**(-2.5)*t4*t5)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 25:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = q**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = CDy*F1*delta*t4*t6
        t8 = t5**(-2.5)
        t9 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t10 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t11 = t10**2
        t12 = 2*t10
        return np.pi**2.5*(-2*ABz*CDy*CDz*F0*alpha*t3*t5**(-0.5)/(p**2*q**3) + 2*ABz*CDz*F1*alpha*gamma*t0*t4*t6*t9 + 2*ABz*F2*alpha*t10*t2*t8*t9 - ABz*t1*t12*t7 + 2*CDy*CDz*F1*delta*gamma*t0*t10*t4*t6 + 2*CDy*F2*delta*t11*t2*t8 - CDz*F2*gamma*t12*t2*t8*t9 + F2*t2*t8*t9 - 2*F3*p*t11*t5**(-3.5)*t9 - t0*t7)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 26:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = q**(-2.0)
        t8 = ABz*t1*t6*t7
        t9 = p**(-2.0)
        t10 = t4**(-0.5)
        t11 = CDy**2
        t12 = t4**(-2.5)
        t13 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t14 = t13*t2
        t15 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t16 = t15**2
        t17 = 2*CDy*t15
        return np.pi**2.5*(2*ABz*CDy*F1*alpha*gamma*t0*t15*t5*t7 + ABz*F0*alpha*t10*t7*t9 - 2*ABz*F0*alpha*t10*t11*t3*t9/q**3 + 2*ABz*F2*alpha*t12*t16*t2 + 2*CDy*F2*delta*t12*t13*t15*t2 + 2*F1*delta*gamma*t0*t11*t13*t5*t7 - F2*gamma*t12*t14*t17 + F2*t12*t13*t2 - 2*F3*p*t13*t16*t4**(-3.5) - delta*t17*t8 - t0*t14*t6 - t8)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 27:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = p + q
        t4 = CDx*gamma
        t5 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t6 = q**(-2.0)
        t7 = t3**(-1.5)
        t8 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t9 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t10 = t3**(-2.5)
        return 2*np.pi**2.5*(ABz*CDx*F1*alpha*gamma*t0*t6*t7*t8 - ABz*CDy*F0*alpha*delta*t3**(-0.5)*t4/(p**2*q**3) - ABz*CDy*F1*delta*t1*t5*t6*t7 + ABz*F2*alpha*t10*t2*t5*t8 + CDx*CDy*F1*delta*gamma*t0*t6*t7*t9 + CDy*F2*delta*t10*t2*t5*t9 - F2*t10*t2*t4*t8*t9 - F3*p*t3**(-3.5)*t5*t8*t9)*np.exp(-beta*rAB*t1 - delta*gamma*rCD*t2)
    if case_id == 28:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = t2**(-1.5)
        t4 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        t5 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        return 2*np.pi**2.5*(-ABz*CDx*F0*alpha*delta*t2**(-0.5)/(p**2*q**2) + ABz*F1*alpha*t0*t1*t3*t4 + CDx*F1*delta*t0*t1*t3*t5 - F2*t2**(-2.5)*t4*t5)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 29:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = q**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = CDx*F1*delta*t4*t6
        t8 = t5**(-2.5)
        t9 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t10 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t11 = t10**2
        t12 = 2*t10
        return np.pi**2.5*(-2*ABz*CDx*CDz*F0*alpha*t3*t5**(-0.5)/(p**2*q**3) + 2*ABz*CDz*F1*alpha*gamma*t0*t4*t6*t9 + 2*ABz*F2*alpha*t10*t2*t8*t9 - ABz*t1*t12*t7 + 2*CDx*CDz*F1*delta*gamma*t0*t10*t4*t6 + 2*CDx*F2*delta*t11*t2*t8 - CDz*F2*gamma*t12*t2*t8*t9 + F2*t2*t8*t9 - 2*F3*p*t11*t5**(-3.5)*t9 - t0*t7)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 30:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = p + q
        t4 = CDy*gamma
        t5 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t6 = q**(-2.0)
        t7 = t3**(-1.5)
        t8 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t9 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t10 = t3**(-2.5)
        return 2*np.pi**2.5*(-ABz*CDx*F0*alpha*delta*t3**(-0.5)*t4/(p**2*q**3) - ABz*CDx*F1*delta*t1*t5*t6*t7 + ABz*CDy*F1*alpha*gamma*t0*t6*t7*t8 + ABz*F2*alpha*t10*t2*t5*t8 + CDx*CDy*F1*delta*gamma*t0*t6*t7*t9 + CDx*F2*delta*t10*t2*t5*t9 - F2*t10*t2*t4*t8*t9 - F3*p*t3**(-3.5)*t5*t8*t9)*np.exp(-beta*rAB*t1 - delta*gamma*rCD*t2)
    if case_id == 31:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = q**(-2.0)
        t8 = ABz*t1*t6*t7
        t9 = p**(-2.0)
        t10 = t4**(-0.5)
        t11 = CDx**2
        t12 = t4**(-2.5)
        t13 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t14 = t13*t2
        t15 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t16 = t15**2
        t17 = 2*CDx*t15
        return np.pi**2.5*(2*ABz*CDx*F1*alpha*gamma*t0*t15*t5*t7 + ABz*F0*alpha*t10*t7*t9 - 2*ABz*F0*alpha*t10*t11*t3*t9/q**3 + 2*ABz*F2*alpha*t12*t16*t2 + 2*CDx*F2*delta*t12*t13*t15*t2 + 2*F1*delta*gamma*t0*t11*t13*t5*t7 - F2*gamma*t12*t14*t17 + F2*t12*t13*t2 - 2*F3*p*t13*t16*t4**(-3.5) - delta*t17*t8 - t0*t14*t6 - t8)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 32:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = p + q
        return 2*np.pi**2.5*t0*(ABy*F0*t1*t2*t3**(-0.5) - F1*t3**(-1.5)*(t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)))*np.exp(-beta*rAB*t1 - delta*gamma*rCD*t2)
    if case_id == 33:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        return 2*np.pi**2.5*(ABy*CDz*F0*alpha*gamma*t4**(-0.5)/(p**2*q**2) + ABy*t1*t2*t5*t6 - CDz*t0*t3*t6*t7 - F2*t4**(-2.5)*t5*t7)*np.exp(-beta*rAB*t1 - delta*rCD*t3)
    if case_id == 34:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = F1*t0*t1*t2**(-1.5)
        t4 = CDy*gamma
        t5 = ABy*alpha
        t6 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        t7 = 2*t3*t6
        return np.pi**2.5*(2*F0*t2**(-0.5)*t4*t5/(p**2*q**2) - 2*F2*t2**(-2.5)*t6**2 + t3 - t4*t7 + t5*t7)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 35:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        return 2*np.pi**2.5*(ABy*CDx*F0*alpha*gamma*t4**(-0.5)/(p**2*q**2) + ABy*t1*t2*t5*t6 - CDx*t0*t3*t6*t7 - F2*t4**(-2.5)*t5*t7)*np.exp(-beta*rAB*t1 - delta*rCD*t3)
    if case_id == 36:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = t2**(-1.5)
        t4 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        t5 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        return 2*np.pi**2.5*(-ABy*CDz*F0*alpha*delta*t2**(-0.5)/(p**2*q**2) + ABy*F1*alpha*t0*t1*t3*t4 + CDz*F1*delta*t0*t1*t3*t5 - F2*t2**(-2.5)*t4*t5)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 37:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = q**(-2.0)
        t8 = ABy*t1*t6*t7
        t9 = p**(-2.0)
        t10 = t4**(-0.5)
        t11 = CDz**2
        t12 = t4**(-2.5)
        t13 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t14 = t13*t2
        t15 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t16 = t15**2
        t17 = 2*CDz*t15
        return np.pi**2.5*(2*ABy*CDz*F1*alpha*gamma*t0*t15*t5*t7 + ABy*F0*alpha*t10*t7*t9 - 2*ABy*F0*alpha*t10*t11*t3*t9/q**3 + 2*ABy*F2*alpha*t12*t16*t2 + 2*CDz*F2*delta*t12*t13*t15*t2 + 2*F1*delta*gamma*t0*t11*t13*t5*t7 - F2*gamma*t12*t14*t17 + F2*t12*t13*t2 - 2*F3*p*t13*t16*t4**(-3.5) - delta*t17*t8 - t0*t14*t6 - t8)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 38:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = q**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = CDz*F1*delta*t4*t6
        t8 = t5**(-2.5)
        t9 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t10 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t11 = t10**2
        t12 = 2*t10
        return np.pi**2.5*(-2*ABy*CDy*CDz*F0*alpha*t3*t5**(-0.5)/(p**2*q**3) + 2*ABy*CDy*F1*alpha*gamma*t0*t4*t6*t9 + 2*ABy*F2*alpha*t10*t2*t8*t9 - ABy*t1*t12*t7 + 2*CDy*CDz*F1*delta*gamma*t0*t10*t4*t6 - CDy*F2*gamma*t12*t2*t8*t9 + 2*CDz*F2*delta*t11*t2*t8 + F2*t2*t8*t9 - 2*F3*p*t11*t5**(-3.5)*t9 - t0*t7)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 39:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = p + q
        t4 = CDx*gamma
        t5 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t6 = q**(-2.0)
        t7 = t3**(-1.5)
        t8 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t9 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t10 = t3**(-2.5)
        return 2*np.pi**2.5*(ABy*CDx*F1*alpha*gamma*t0*t6*t7*t8 - ABy*CDz*F0*alpha*delta*t3**(-0.5)*t4/(p**2*q**3) - ABy*CDz*F1*delta*t1*t5*t6*t7 + ABy*F2*alpha*t10*t2*t5*t8 + CDx*CDz*F1*delta*gamma*t0*t6*t7*t9 + CDz*F2*delta*t10*t2*t5*t9 - F2*t10*t2*t4*t8*t9 - F3*p*t3**(-3.5)*t5*t8*t9)*np.exp(-beta*rAB*t1 - delta*gamma*rCD*t2)
    if case_id == 40:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = F1*t0*t1*t2**(-1.5)
        t4 = ABy*alpha
        t5 = CDy*delta
        t6 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        t7 = 2*t3*t6
        return np.pi**2.5*(-2*F0*t2**(-0.5)*t4*t5/(p**2*q**2) - 2*F2*t2**(-2.5)*t6**2 + t3 + t4*t7 + t5*t7)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 41:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDz*gamma
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABy
        t9 = alpha*t8
        t10 = CDy*CDz*t3
        t11 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t15 = 2*t14**2
        t16 = CDy*delta
        t17 = t1*t8
        t18 = t13*t14
        return np.pi**2.5*(-F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 + t0*t7 - t11*t16*t17*t6 - t12*t15*t4 + t13 + t14*t17*t7 + 2*t16*t18 + t18*t9)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 42:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = q**(-2.0)
        t8 = ABy*t1*t6*t7
        t9 = t0*t6
        t10 = CDy*delta
        t11 = p**(-2.0)
        t12 = t4**(-0.5)
        t13 = CDy**2
        t14 = t4**(-2.5)
        t15 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t16 = t15**2
        return np.pi**2.5*(2*ABy*CDy*F1*alpha*gamma*t0*t15*t5*t7 - 2*ABy*F0*alpha*delta*gamma*t11*t12*t13/q**3 + ABy*F0*alpha*t11*t12*t7 + 2*ABy*F2*alpha*t14*t16*t2 + CDy*F1*gamma*t0*t5*t7 + 2*CDy*F2*delta*t14*t16*t2 - 2*CDy*F2*t14*t16*t3 + 2*F1*delta*gamma*t0*t13*t15*t5*t7 + 3*F2*t14*t15*t2 - 2*F3*p*t15**3*t4**(-3.5) - 2*t10*t15*t8 - t10*t7*t9 - t15*t2*t9 - t8)*np.exp(-beta*rAB*t1 - delta*rCD*t3)
    if case_id == 43:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDx*gamma
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABy
        t9 = alpha*t8
        t10 = CDx*CDy*t3
        t11 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t15 = 2*t14**2
        t16 = t1*t8
        t17 = CDy*delta
        t18 = t13*t14
        return np.pi**2.5*(-F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 + t0*t7 - t11*t16*t17*t6 - t12*t15*t4 + t13 + t14*t16*t7 + 2*t17*t18 + t18*t9)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 44:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = t2**(-1.5)
        t4 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        t5 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        return 2*np.pi**2.5*(-ABy*CDx*F0*alpha*delta*t2**(-0.5)/(p**2*q**2) + ABy*F1*alpha*t0*t1*t3*t4 + CDx*F1*delta*t0*t1*t3*t5 - F2*t2**(-2.5)*t4*t5)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 45:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = p + q
        t4 = CDz*gamma
        t5 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t6 = q**(-2.0)
        t7 = t3**(-1.5)
        t8 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t9 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t10 = t3**(-2.5)
        return 2*np.pi**2.5*(-ABy*CDx*F0*alpha*delta*t3**(-0.5)*t4/(p**2*q**3) - ABy*CDx*F1*delta*t1*t5*t6*t7 + ABy*CDz*F1*alpha*gamma*t0*t6*t7*t8 + ABy*F2*alpha*t10*t2*t5*t8 + CDx*CDz*F1*delta*gamma*t0*t6*t7*t9 + CDx*F2*delta*t10*t2*t5*t9 - F2*t10*t2*t4*t8*t9 - F3*p*t3**(-3.5)*t5*t8*t9)*np.exp(-beta*rAB*t1 - delta*gamma*rCD*t2)
    if case_id == 46:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = q**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = CDx*F1*delta*t4*t6
        t8 = t5**(-2.5)
        t9 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t10 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t11 = t10**2
        t12 = 2*t10
        return np.pi**2.5*(-2*ABy*CDx*CDy*F0*alpha*t3*t5**(-0.5)/(p**2*q**3) + 2*ABy*CDy*F1*alpha*gamma*t0*t4*t6*t9 + 2*ABy*F2*alpha*t10*t2*t8*t9 - ABy*t1*t12*t7 + 2*CDx*CDy*F1*delta*gamma*t0*t10*t4*t6 + 2*CDx*F2*delta*t11*t2*t8 - CDy*F2*gamma*t12*t2*t8*t9 + F2*t2*t8*t9 - 2*F3*p*t11*t5**(-3.5)*t9 - t0*t7)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 47:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = q**(-2.0)
        t8 = ABy*t1*t6*t7
        t9 = p**(-2.0)
        t10 = t4**(-0.5)
        t11 = CDx**2
        t12 = t4**(-2.5)
        t13 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t14 = t13*t2
        t15 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t16 = t15**2
        t17 = 2*CDx*t15
        return np.pi**2.5*(2*ABy*CDx*F1*alpha*gamma*t0*t15*t5*t7 + ABy*F0*alpha*t10*t7*t9 - 2*ABy*F0*alpha*t10*t11*t3*t9/q**3 + 2*ABy*F2*alpha*t12*t16*t2 + 2*CDx*F2*delta*t12*t13*t15*t2 + 2*F1*delta*gamma*t0*t11*t13*t5*t7 - F2*gamma*t12*t14*t17 + F2*t12*t13*t2 - 2*F3*p*t13*t16*t4**(-3.5) - delta*t17*t8 - t0*t14*t6 - t8)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 48:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = p + q
        return 2*np.pi**2.5*t0*(ABx*F0*t1*t2*t3**(-0.5) - F1*t3**(-1.5)*(t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)))*np.exp(-beta*rAB*t1 - delta*gamma*rCD*t2)
    if case_id == 49:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        return 2*np.pi**2.5*(ABx*CDz*F0*alpha*gamma*t4**(-0.5)/(p**2*q**2) + ABx*t1*t2*t5*t6 - CDz*t0*t3*t6*t7 - F2*t4**(-2.5)*t5*t7)*np.exp(-beta*rAB*t1 - delta*rCD*t3)
    if case_id == 50:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        return 2*np.pi**2.5*(ABx*CDy*F0*alpha*gamma*t4**(-0.5)/(p**2*q**2) + ABx*t1*t2*t5*t6 - CDy*t0*t3*t6*t7 - F2*t4**(-2.5)*t5*t7)*np.exp(-beta*rAB*t1 - delta*rCD*t3)
    if case_id == 51:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = F1*t0*t1*t2**(-1.5)
        t4 = CDx*gamma
        t5 = ABx*alpha
        t6 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        t7 = 2*t3*t6
        return np.pi**2.5*(2*F0*t2**(-0.5)*t4*t5/(p**2*q**2) - 2*F2*t2**(-2.5)*t6**2 + t3 - t4*t7 + t5*t7)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 52:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = t2**(-1.5)
        t4 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        t5 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        return 2*np.pi**2.5*(-ABx*CDz*F0*alpha*delta*t2**(-0.5)/(p**2*q**2) + ABx*F1*alpha*t0*t1*t3*t4 + CDz*F1*delta*t0*t1*t3*t5 - F2*t2**(-2.5)*t4*t5)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 53:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = q**(-2.0)
        t8 = ABx*t1*t6*t7
        t9 = p**(-2.0)
        t10 = t4**(-0.5)
        t11 = CDz**2
        t12 = t4**(-2.5)
        t13 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t14 = t13*t2
        t15 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t16 = t15**2
        t17 = 2*CDz*t15
        return np.pi**2.5*(2*ABx*CDz*F1*alpha*gamma*t0*t15*t5*t7 + ABx*F0*alpha*t10*t7*t9 - 2*ABx*F0*alpha*t10*t11*t3*t9/q**3 + 2*ABx*F2*alpha*t12*t16*t2 + 2*CDz*F2*delta*t12*t13*t15*t2 + 2*F1*delta*gamma*t0*t11*t13*t5*t7 - F2*gamma*t12*t14*t17 + F2*t12*t13*t2 - 2*F3*p*t13*t16*t4**(-3.5) - delta*t17*t8 - t0*t14*t6 - t8)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 54:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = p + q
        t4 = CDy*gamma
        t5 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t6 = q**(-2.0)
        t7 = t3**(-1.5)
        t8 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t9 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t10 = t3**(-2.5)
        return 2*np.pi**2.5*(ABx*CDy*F1*alpha*gamma*t0*t6*t7*t8 - ABx*CDz*F0*alpha*delta*t3**(-0.5)*t4/(p**2*q**3) - ABx*CDz*F1*delta*t1*t5*t6*t7 + ABx*F2*alpha*t10*t2*t5*t8 + CDy*CDz*F1*delta*gamma*t0*t6*t7*t9 + CDz*F2*delta*t10*t2*t5*t9 - F2*t10*t2*t4*t8*t9 - F3*p*t3**(-3.5)*t5*t8*t9)*np.exp(-beta*rAB*t1 - delta*gamma*rCD*t2)
    if case_id == 55:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = q**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = CDz*F1*delta*t4*t6
        t8 = t5**(-2.5)
        t9 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t10 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t11 = t10**2
        t12 = 2*t10
        return np.pi**2.5*(-2*ABx*CDx*CDz*F0*alpha*t3*t5**(-0.5)/(p**2*q**3) + 2*ABx*CDx*F1*alpha*gamma*t0*t4*t6*t9 + 2*ABx*F2*alpha*t10*t2*t8*t9 - ABx*t1*t12*t7 + 2*CDx*CDz*F1*delta*gamma*t0*t10*t4*t6 - CDx*F2*gamma*t12*t2*t8*t9 + 2*CDz*F2*delta*t11*t2*t8 + F2*t2*t8*t9 - 2*F3*p*t11*t5**(-3.5)*t9 - t0*t7)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 56:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = t2**(-1.5)
        t4 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        t5 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        return 2*np.pi**2.5*(-ABx*CDy*F0*alpha*delta*t2**(-0.5)/(p**2*q**2) + ABx*F1*alpha*t0*t1*t3*t4 + CDy*F1*delta*t0*t1*t3*t5 - F2*t2**(-2.5)*t4*t5)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 57:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = p + q
        t4 = CDz*gamma
        t5 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t6 = q**(-2.0)
        t7 = t3**(-1.5)
        t8 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t9 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t10 = t3**(-2.5)
        return 2*np.pi**2.5*(-ABx*CDy*F0*alpha*delta*t3**(-0.5)*t4/(p**2*q**3) - ABx*CDy*F1*delta*t1*t5*t6*t7 + ABx*CDz*F1*alpha*gamma*t0*t6*t7*t8 + ABx*F2*alpha*t10*t2*t5*t8 + CDy*CDz*F1*delta*gamma*t0*t6*t7*t9 + CDy*F2*delta*t10*t2*t5*t9 - F2*t10*t2*t4*t8*t9 - F3*p*t3**(-3.5)*t5*t8*t9)*np.exp(-beta*rAB*t1 - delta*gamma*rCD*t2)
    if case_id == 58:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = q**(-2.0)
        t8 = ABx*t1*t6*t7
        t9 = p**(-2.0)
        t10 = t4**(-0.5)
        t11 = CDy**2
        t12 = t4**(-2.5)
        t13 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t14 = t13*t2
        t15 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t16 = t15**2
        t17 = 2*CDy*t15
        return np.pi**2.5*(2*ABx*CDy*F1*alpha*gamma*t0*t15*t5*t7 + ABx*F0*alpha*t10*t7*t9 - 2*ABx*F0*alpha*t10*t11*t3*t9/q**3 + 2*ABx*F2*alpha*t12*t16*t2 + 2*CDy*F2*delta*t12*t13*t15*t2 + 2*F1*delta*gamma*t0*t11*t13*t5*t7 - F2*gamma*t12*t14*t17 + F2*t12*t13*t2 - 2*F3*p*t13*t16*t4**(-3.5) - delta*t17*t8 - t0*t14*t6 - t8)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 59:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = q**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = CDy*F1*delta*t4*t6
        t8 = t5**(-2.5)
        t9 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t10 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t11 = t10**2
        t12 = 2*t10
        return np.pi**2.5*(-2*ABx*CDx*CDy*F0*alpha*t3*t5**(-0.5)/(p**2*q**3) + 2*ABx*CDx*F1*alpha*gamma*t0*t4*t6*t9 + 2*ABx*F2*alpha*t10*t2*t8*t9 - ABx*t1*t12*t7 + 2*CDx*CDy*F1*delta*gamma*t0*t10*t4*t6 - CDx*F2*gamma*t12*t2*t8*t9 + 2*CDy*F2*delta*t11*t2*t8 + F2*t2*t8*t9 - 2*F3*p*t11*t5**(-3.5)*t9 - t0*t7)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 60:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = F1*t0*t1*t2**(-1.5)
        t4 = ABx*alpha
        t5 = CDx*delta
        t6 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        t7 = 2*t3*t6
        return np.pi**2.5*(-2*F0*t2**(-0.5)*t4*t5/(p**2*q**2) - 2*F2*t2**(-2.5)*t6**2 + t3 + t4*t7 + t5*t7)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 61:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDz*gamma
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABx
        t9 = alpha*t8
        t10 = CDx*CDz*t3
        t11 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t15 = 2*t14**2
        t16 = CDx*delta
        t17 = t1*t8
        t18 = t13*t14
        return np.pi**2.5*(-F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 + t0*t7 - t11*t16*t17*t6 - t12*t15*t4 + t13 + t14*t17*t7 + 2*t16*t18 + t18*t9)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 62:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDy*gamma
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABx
        t9 = alpha*t8
        t10 = CDx*CDy*t3
        t11 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t15 = 2*t14**2
        t16 = CDx*delta
        t17 = t1*t8
        t18 = t13*t14
        return np.pi**2.5*(-F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 + t0*t7 - t11*t16*t17*t6 - t12*t15*t4 + t13 + t14*t17*t7 + 2*t16*t18 + t18*t9)*np.exp(-beta*rAB*t1 - rCD*t2*t3)
    if case_id == 63:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = q**(-2.0)
        t8 = ABx*t1*t6*t7
        t9 = t0*t6
        t10 = CDx*delta
        t11 = p**(-2.0)
        t12 = t4**(-0.5)
        t13 = CDx**2
        t14 = t4**(-2.5)
        t15 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t16 = t15**2
        return np.pi**2.5*(2*ABx*CDx*F1*alpha*gamma*t0*t15*t5*t7 - 2*ABx*F0*alpha*delta*gamma*t11*t12*t13/q**3 + ABx*F0*alpha*t11*t12*t7 + 2*ABx*F2*alpha*t14*t16*t2 + CDx*F1*gamma*t0*t5*t7 + 2*CDx*F2*delta*t14*t16*t2 - 2*CDx*F2*t14*t16*t3 + 2*F1*delta*gamma*t0*t13*t15*t5*t7 + 3*F2*t14*t15*t2 - 2*F3*p*t15**3*t4**(-3.5) - 2*t10*t15*t8 - t10*t7*t9 - t15*t2*t9 - t8)*np.exp(-beta*rAB*t1 - delta*rCD*t3)
    if case_id == 64:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = p + q
        return -2*np.pi**2.5*t0*(ABz*F0*t1*t2*t3**(-0.5) + F1*t3**(-1.5)*(t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)))*np.exp(-alpha*rAB*t1 - delta*gamma*rCD*t2)
    if case_id == 65:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = t2**(-1.5)
        t4 = ABz*beta
        t5 = CDz*gamma
        t6 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        t7 = 2*F1*t0*t1*t3*t6
        return np.pi**2.5*(-2*F0*t2**(-0.5)*t4*t5/(p**2*q**2) + F1*t0*t1*t3 - 2*F2*t2**(-2.5)*t6**2 - t4*t7 - t5*t7)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 66:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        return -2*np.pi**2.5*(ABz*CDy*F0*beta*gamma*t4**(-0.5)/(p**2*q**2) + ABz*t1*t2*t5*t6 + CDy*t0*t3*t6*t7 + F2*t4**(-2.5)*t5*t7)*np.exp(-alpha*rAB*t1 - delta*rCD*t3)
    if case_id == 67:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        return -2*np.pi**2.5*(ABz*CDx*F0*beta*gamma*t4**(-0.5)/(p**2*q**2) + ABz*t1*t2*t5*t6 + CDx*t0*t3*t6*t7 + F2*t4**(-2.5)*t5*t7)*np.exp(-alpha*rAB*t1 - delta*rCD*t3)
    if case_id == 68:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = F1*t0*t1*t2**(-1.5)
        t4 = ABz*beta
        t5 = CDz*delta
        t6 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        t7 = 2*t3*t6
        return np.pi**2.5*(2*F0*t2**(-0.5)*t4*t5/(p**2*q**2) - 2*F2*t2**(-2.5)*t6**2 + t3 - t4*t7 + t5*t7)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 69:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = q**(-2.0)
        t7 = ABz*t6
        t8 = t1*t5*t7
        t9 = CDz*gamma
        t10 = t0*t5
        t11 = t10*t6
        t12 = F0*beta*t4**(-0.5)/p**2
        t13 = CDz*delta
        t14 = 2*ABz
        t15 = CDz**2*delta*gamma
        t16 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t17 = t16*t2
        t18 = F2*t4**(-2.5)
        t19 = t16**2*t18
        t20 = t19*t2
        t21 = 2*t13
        t22 = 2*t16
        return np.pi**2.5*(-2*CDz*t19*t3 - 2*F3*p*t16**3*t4**(-3.5) - beta*t14*t20 - t10*t17 - t11*t13 + t11*t15*t22 + t11*t9 - t12*t7 + t16*t21*t8 + 3*t17*t18 + t20*t21 - t22*t8*t9 + t8 + t12*t14*t15/q**3)*np.exp(-alpha*rAB*t1 - delta*rCD*t3)
    if case_id == 70:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDy*gamma
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABz
        t9 = beta*t8
        t10 = CDy*CDz*t3
        t11 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t15 = 2*t14**2
        t16 = t1*t8
        t17 = CDz*delta
        t18 = t13*t14
        return np.pi**2.5*(F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 + t0*t7 + t11*t16*t17*t6 - t12*t15*t4 + t13 - t14*t16*t7 + 2*t17*t18 - t18*t9)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 71:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDx*gamma
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABz
        t9 = beta*t8
        t10 = CDx*CDz*t3
        t11 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t15 = 2*t14**2
        t16 = t1*t8
        t17 = CDz*delta
        t18 = t13*t14
        return np.pi**2.5*(F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 + t0*t7 + t11*t16*t17*t6 - t12*t15*t4 + t13 - t14*t16*t7 + 2*t17*t18 - t18*t9)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 72:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        return 2*np.pi**2.5*(ABz*CDy*F0*beta*delta*t4**(-0.5)/(p**2*q**2) - ABz*t1*t2*t6*t7 + CDy*t0*t3*t5*t6 - F2*t4**(-2.5)*t5*t7)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 73:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDy*delta
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABz
        t9 = beta*t8
        t10 = CDy*CDz*t3
        t11 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t15 = 2*t14**2
        t16 = t1*t8
        t17 = CDz*gamma
        t18 = t13*t14
        return np.pi**2.5*(F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 - t0*t7 - t11*t16*t17*t6 + t12*t15*t4 + t13 + t14*t16*t7 - 2*t17*t18 - t18*t9)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 74:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = q**(-2.0)
        t7 = ABz*t6
        t8 = t1*t5*t7
        t9 = F0*beta*t4**(-0.5)/p**2
        t10 = 2*CDy**2*t3
        t11 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t12 = t11*t2
        t13 = F2*t4**(-2.5)
        t14 = t12*t13
        t15 = t0*t5
        t16 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t17 = 2*t16**2
        t18 = 2*CDy*t16
        t19 = t18*t8
        t20 = t14*t18
        return np.pi**2.5*(-ABz*beta*t13*t17*t2 + ABz*t10*t9/q**3 - F3*p*t11*t17*t4**(-3.5) + delta*t19 + delta*t20 - gamma*t19 - gamma*t20 + t10*t11*t15*t6 - t12*t15 + t14 - t7*t9 + t8)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 75:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = CDy*delta
        t6 = CDx*gamma
        t7 = ABz*t6
        t8 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t9 = ABz*t8
        t10 = q**(-2.0)
        t11 = t4**(-1.5)
        t12 = F1*t1*t10*t11
        t13 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t14 = t13*t6
        t15 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t16 = t13*t8
        t17 = F2*t4**(-2.5)
        t18 = t15*t17*t2
        return 2*np.pi**2.5*(CDy*t16*t17*t3 + F0*beta*t4**(-0.5)*t5*t7/(p**2*q**3) + F1*t0*t10*t11*t14*t5 - F3*p*t15*t16*t4**(-3.5) - beta*t18*t9 - t12*t15*t7 + t12*t5*t9 - t14*t18)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 76:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        return 2*np.pi**2.5*(ABz*CDx*F0*beta*delta*t4**(-0.5)/(p**2*q**2) - ABz*t1*t2*t6*t7 + CDx*t0*t3*t5*t6 - F2*t4**(-2.5)*t5*t7)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 77:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDx*delta
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABz
        t9 = beta*t8
        t10 = CDx*CDz*t3
        t11 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t15 = 2*t14**2
        t16 = t1*t8
        t17 = CDz*gamma
        t18 = t13*t14
        return np.pi**2.5*(F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 - t0*t7 - t11*t16*t17*t6 + t12*t15*t4 + t13 + t14*t16*t7 - 2*t17*t18 - t18*t9)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 78:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = CDx*delta
        t6 = CDy*gamma
        t7 = ABz*t6
        t8 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t9 = ABz*t8
        t10 = q**(-2.0)
        t11 = t4**(-1.5)
        t12 = F1*t1*t10*t11
        t13 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t14 = t13*t6
        t15 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t16 = t13*t8
        t17 = F2*t4**(-2.5)
        t18 = t15*t17*t2
        return 2*np.pi**2.5*(CDx*t16*t17*t3 + F0*beta*t4**(-0.5)*t5*t7/(p**2*q**3) + F1*t0*t10*t11*t14*t5 - F3*p*t15*t16*t4**(-3.5) - beta*t18*t9 - t12*t15*t7 + t12*t5*t9 - t14*t18)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 79:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = q**(-2.0)
        t7 = ABz*t6
        t8 = t1*t5*t7
        t9 = F0*beta*t4**(-0.5)/p**2
        t10 = 2*CDx**2*t3
        t11 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t12 = t11*t2
        t13 = F2*t4**(-2.5)
        t14 = t12*t13
        t15 = t0*t5
        t16 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t17 = 2*t16**2
        t18 = 2*CDx*t16
        t19 = t18*t8
        t20 = t14*t18
        return np.pi**2.5*(-ABz*beta*t13*t17*t2 + ABz*t10*t9/q**3 - F3*p*t11*t17*t4**(-3.5) + delta*t19 + delta*t20 - gamma*t19 - gamma*t20 + t10*t11*t15*t6 - t12*t15 + t14 - t7*t9 + t8)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 80:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = p + q
        t4 = t3**(-1.5)
        t5 = F1*t0*t4
        t6 = t3**(-0.5)
        t7 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        return np.pi**2.5*t0*(-2*ABz**2*F0*t1*t2*t6/p**2 + 2*ABz*F1*beta*t0*t4*t7 - 2*ABz*alpha*t5*t7 + F0*t0*t2*t6 + 2*F2*q*t3**(-2.5)*t7**2 - t5)*np.exp(-delta*gamma*rCD*t2 - rAB*t0*t1)
    if case_id == 81:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = t2*t6
        t8 = p**(-2.0)
        t9 = ABz*t8
        t10 = t3*t6
        t11 = q**(-2.0)
        t12 = t4**(-0.5)
        t13 = ABz**2*beta
        t14 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t15 = t4**(-2.5)
        t16 = F2*t15
        t17 = t14**2
        t18 = 2*alpha*t14
        return np.pi**2.5*(2*ABz*CDz*F1*beta*gamma*t14*t2*t5*t8 + ABz*F1*alpha*t2*t5*t8 + 2*ABz*F2*beta*t0*t15*t17 - 2*ABz*t1*t16*t17 - 2*CDz*F0*alpha*gamma*t11*t12*t13/p**3 + CDz*F0*gamma*t11*t12*t8 + 2*CDz*F2*gamma*t0*t15*t17 - CDz*t10*t18*t9 - CDz*t10*t8 + F1*t0*t14*t2*t5 + 2*F3*q*t14**3*t4**(-3.5) - beta*t7*t9 - 3*t0*t14*t16 - t13*t18*t7*t8)*np.exp(-beta*rAB*t1 - delta*rCD*t3)
    if case_id == 82:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = p**(-2.0)
        t8 = CDy*t3*t6*t7
        t9 = q**(-2.0)
        t10 = t4**(-0.5)
        t11 = 2*ABz**2*t1
        t12 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t13 = t4**(-2.5)
        t14 = F2*t0*t12*t13
        t15 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t16 = t15**2
        t17 = 2*ABz*t15
        return np.pi**2.5*(2*ABz*CDy*F1*beta*gamma*t15*t2*t5*t7 + 2*ABz*F2*beta*t0*t12*t13*t15 + CDy*F0*gamma*t10*t7*t9 - CDy*F0*gamma*t10*t11*t9/p**3 + 2*CDy*F2*gamma*t0*t13*t16 + F1*t0*t12*t2*t5 + 2*F3*q*t12*t16*t4**(-3.5) - alpha*t14*t17 - alpha*t17*t8 - t11*t12*t2*t6*t7 - t14 - t8)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 83:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = p**(-2.0)
        t8 = CDx*t3*t6*t7
        t9 = q**(-2.0)
        t10 = t4**(-0.5)
        t11 = 2*ABz**2*t1
        t12 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t13 = t4**(-2.5)
        t14 = F2*t0*t12*t13
        t15 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t16 = t15**2
        t17 = 2*ABz*t15
        return np.pi**2.5*(2*ABz*CDx*F1*beta*gamma*t15*t2*t5*t7 + 2*ABz*F2*beta*t0*t12*t13*t15 + CDx*F0*gamma*t10*t7*t9 - CDx*F0*gamma*t10*t11*t9/p**3 + 2*CDx*F2*gamma*t0*t13*t16 + F1*t0*t12*t2*t5 + 2*F3*q*t12*t16*t4**(-3.5) - alpha*t14*t17 - alpha*t17*t8 - t11*t12*t2*t6*t7 - t14 - t8)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 84:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = t2*t5
        t7 = p**(-2.0)
        t8 = ABz*t7
        t9 = t6*t8
        t10 = CDz*t7
        t11 = t3*t5
        t12 = F0*delta*t4**(-0.5)/q**2
        t13 = ABz**2*alpha
        t14 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t15 = t0*t14
        t16 = F2*t4**(-2.5)
        t17 = 2*t14**2*t16
        t18 = ABz*t17
        t19 = CDz*t11*t8
        t20 = 2*beta*t14
        return np.pi**2.5*(2*CDz*beta*t12*t13/p**3 - CDz*delta*t0*t17 + 2*F3*q*t14**3*t4**(-3.5) + 2*alpha*t14*t19 + alpha*t9 + beta*t0*t18 - beta*t9 - t1*t18 + t10*t11 - t10*t12 - t13*t20*t6*t7 - 3*t15*t16 + t15*t6 - t19*t20)*np.exp(-beta*rAB*t1 - gamma*rCD*t3)
    if case_id == 85:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = t2*t3
        t5 = q**(-2.0)
        t6 = (1/2)*t5
        t7 = p**(-2.0)
        t8 = p + q
        t9 = F0*t8**(-0.5)
        t10 = t7*t9
        t11 = F1*t8**(-1.5)
        t12 = t11*t7
        t13 = t12*t2
        t14 = t0*t11
        t15 = t8**(-2.5)
        t16 = F2*t0*t15*t2
        t17 = ABz*alpha
        t18 = t12*t5
        t19 = CDz*t18
        t20 = t17*t19
        t21 = ABz*beta
        t22 = t19*t21
        t23 = ABz**2*t1
        t24 = t18*t23
        t25 = CDz**2
        t26 = t25*t3
        t27 = t18*t26
        t28 = t23*t9/p**3
        t29 = t26/q**3
        t30 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t31 = t30**2
        t32 = F3*t8**(-3.5)
        t33 = F2*t15*t31
        t34 = t0*t33
        t35 = 2*t30**3*t32
        t36 = CDz*t35
        t37 = t13*t30
        t38 = CDz*t30
        t39 = gamma*t38
        t40 = t14*t5
        t41 = 3*t16
        t42 = t17*t30
        t43 = t21*t30
        t44 = delta*t38
        t45 = 2*t34
        t46 = t2*t45
        t47 = CDz*t46
        t48 = t17*t47
        t49 = t21*t47
        t50 = 2*t27
        t51 = 2*t24
        return np.pi**2.5*(2*F4*p*q*t30**4*t8**(-4.5) - delta*t20 + delta*t22 - delta*t36 + delta*t48 - delta*t49 + gamma*t20 - gamma*t22 + gamma*t36 - gamma*t48 + gamma*t49 - t10*t29 + t10*t6 - 1/2*t13 - t14*t6 + (3/2)*t16 - t17*t35 - t17*t37 + t2*t33 + t21*t35 + t21*t37 - t23*t46 + t24 - t25*t4*t45 + t27 + 2*t28*t29 - t28*t5 - 6*t31*t32 + t34 + t39*t40 - t39*t41 - t39*t51 - t40*t44 + t41*t42 - t41*t43 + t41*t44 + t42*t50 - t43*t50 + t44*t51)*np.exp(-rAB*t0*t1 - rCD*t4)
    if case_id == 86:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/q**2
        t8 = p**(-2.0)
        t9 = CDy*gamma
        t10 = t8*t9
        t11 = t10*t7
        t12 = ABz*t11
        t13 = alpha*t12
        t14 = CDz*delta
        t15 = beta*t12
        t16 = F0*t14*t6**(-0.5)/q**3
        t17 = 2*ABz**2
        t18 = alpha*beta*t17
        t19 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t20 = t19**3
        t21 = F3*t6**(-3.5)
        t22 = 2*t21
        t23 = F2*t6**(-2.5)
        t24 = t23*t3
        t25 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t26 = ABz*t25
        t27 = t24*t26
        t28 = t0*t9
        t29 = t19*t28
        t30 = CDz*t25
        t31 = t0*t23
        t32 = t31*t4
        t33 = delta*t30
        t34 = t33*t7
        t35 = ABz*t1
        t36 = t19**2
        t37 = 2*t36
        t38 = t24*t37
        t39 = ABz*beta
        t40 = 2*t14*t19
        t41 = t19*t25
        t42 = t24*t41
        t43 = t22*t36
        t44 = t26*t43
        t45 = 2*CDz*t41
        return np.pi**2.5*(-CDy*CDz*t31*t37*t5 + 2*F4*p*q*t20*t25*t6**(-4.5) - alpha*t44 - beta*t0*t27 + beta*t44 - t0*t34 + t1*t27 - t10*t16 + t11*t14 - t11*t18*t19 + t13*t40 + t13 - t15*t40 - t15 - t17*t2*t42 + t18*t34*t8 + t20*t22*t9 - 3*t21*t41 + t23*t35*t4*t45 - 3*t24*t29 + t28*t38*t39 + t29*t7 + t30*t32 - t32*t39*t45 - t33*t43 - t35*t38*t9 + t42 + t16*t18*t9/p**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 87:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/q**2
        t8 = p**(-2.0)
        t9 = CDx*gamma
        t10 = t8*t9
        t11 = t10*t7
        t12 = ABz*t11
        t13 = alpha*t12
        t14 = CDz*delta
        t15 = beta*t12
        t16 = F0*t14*t6**(-0.5)/q**3
        t17 = 2*ABz**2
        t18 = alpha*beta*t17
        t19 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t20 = t19**3
        t21 = F3*t6**(-3.5)
        t22 = 2*t21
        t23 = F2*t6**(-2.5)
        t24 = t23*t3
        t25 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t26 = ABz*t25
        t27 = t24*t26
        t28 = t0*t9
        t29 = t19*t28
        t30 = CDz*t25
        t31 = t0*t23
        t32 = t31*t4
        t33 = delta*t30
        t34 = t33*t7
        t35 = ABz*t1
        t36 = t19**2
        t37 = 2*t36
        t38 = t24*t37
        t39 = ABz*beta
        t40 = 2*t14*t19
        t41 = t19*t25
        t42 = t24*t41
        t43 = t22*t36
        t44 = t26*t43
        t45 = 2*CDz*t41
        return np.pi**2.5*(-CDx*CDz*t31*t37*t5 + 2*F4*p*q*t20*t25*t6**(-4.5) - alpha*t44 - beta*t0*t27 + beta*t44 - t0*t34 + t1*t27 - t10*t16 + t11*t14 - t11*t18*t19 + t13*t40 + t13 - t15*t40 - t15 - t17*t2*t42 + t18*t34*t8 + t20*t22*t9 - 3*t21*t41 + t23*t35*t4*t45 - 3*t24*t29 + t28*t38*t39 + t29*t7 + t30*t32 - t32*t39*t45 - t33*t43 - t35*t38*t9 + t42 + t16*t18*t9/p**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 88:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = p**(-2.0)
        t7 = CDy*t6
        t8 = t3*t5*t7
        t9 = F0*delta*t4**(-0.5)/q**2
        t10 = 2*ABz**2*t1
        t11 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t12 = t0*t11
        t13 = F2*t4**(-2.5)
        t14 = t12*t13
        t15 = t2*t5
        t16 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t17 = 2*t16**2
        t18 = 2*ABz*t16
        t19 = t18*t8
        t20 = t14*t18
        return np.pi**2.5*(-CDy*delta*t0*t13*t17 + CDy*t10*t9/p**3 + F3*q*t11*t17*t4**(-3.5) + alpha*t19 - alpha*t20 - beta*t19 + beta*t20 - t10*t11*t15*t6 + t12*t15 - t14 - t7*t9 + t8)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 89:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/q**2
        t8 = p**(-2.0)
        t9 = CDy*delta
        t10 = t8*t9
        t11 = t10*t7
        t12 = ABz*t11
        t13 = CDz*gamma
        t14 = F0*t13*t6**(-0.5)/q**3
        t15 = 2*ABz**2
        t16 = alpha*t15
        t17 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t18 = t17**3
        t19 = F3*t6**(-3.5)
        t20 = 2*t19
        t21 = F2*t6**(-2.5)
        t22 = t21*t3
        t23 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t24 = ABz*t23
        t25 = t22*t24
        t26 = CDz*t23
        t27 = gamma*t26
        t28 = t0*t7
        t29 = beta*t0
        t30 = t17*t9
        t31 = t0*t21
        t32 = t17**2
        t33 = 2*t32
        t34 = t22*t33*t9
        t35 = ABz*t1
        t36 = ABz*t29
        t37 = 2*ABz*t13*t30
        t38 = beta*t7*t8
        t39 = t16*t38
        t40 = t17*t23
        t41 = t22*t40
        t42 = t20*t32
        t43 = t24*t42
        t44 = 2*CDz*t21*t4*t40
        return np.pi**2.5*(-CDy*CDz*t31*t33*t5 + 2*F4*p*q*t18*t23*t6**(-4.5) - alpha*t12 + alpha*t37*t7*t8 - alpha*t43 + beta*t12 + beta*t43 + beta*t14*t16*t9/p**3 + 3*t0*t22*t30 + t1*t25 - t10*t14 + t11*t13 - t15*t2*t41 - t18*t20*t9 - 3*t19*t40 - t25*t29 - t26*t31*t4 + t27*t28 - t27*t39 + t27*t42 - t28*t30 + t30*t39 + t34*t35 - t34*t36 - t35*t44 + t36*t44 - t37*t38 + t41)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 90:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = gamma*t3
        t5 = q**(-2.0)
        t6 = (1/2)*t5
        t7 = p**(-2.0)
        t8 = p + q
        t9 = F0*t8**(-0.5)
        t10 = t7*t9
        t11 = (1/2)*t2
        t12 = F1*t8**(-1.5)
        t13 = t12*t7
        t14 = t0*t12
        t15 = F2*t8**(-2.5)
        t16 = t0*t15
        t17 = ABz**2*t1
        t18 = t17*t5
        t19 = t13*t18
        t20 = CDy**2
        t21 = delta*t20
        t22 = gamma*t5
        t23 = t13*t22
        t24 = t9/p**3
        t25 = gamma/q**3
        t26 = 2*delta
        t27 = t20*t26
        t28 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t29 = t28**2
        t30 = F3*t8**(-3.5)
        t31 = t29*t30
        t32 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t33 = t32**2
        t34 = t30*t33
        t35 = t16*t33
        t36 = t2*t29
        t37 = ABz*t32
        t38 = beta*t37
        t39 = t13*t2
        t40 = alpha*t37
        t41 = t16*t2
        t42 = t40*t41
        t43 = CDy*t28
        t44 = t14*t43
        t45 = t16*t3*t43
        t46 = t38*t41
        t47 = t23*t27
        t48 = 2*gamma
        t49 = t43*t48
        t50 = 2*t31
        t51 = t34*t43
        t52 = 2*t45
        return np.pi**2.5*(2*F4*p*q*t29*t33*t8**(-4.5) - delta*t44*t5 - gamma*t41*t43 - t10*t21*t25 + t10*t6 - t11*t13 + t11*t16 - t14*t6 + t15*t36 - 2*t16*t17*t36 + t17*t24*t25*t27 - t18*t24 + t19*t26*t43 - t19*t49 + t19 - 2*t20*t35*t4 + t21*t23 + t22*t44 - t26*t51 - t31 - t34 + t35 + t38*t39 - t38*t47 + t38*t50 - t38*t52 - t39*t40 + t40*t47 - t40*t50 + t40*t52 - t42*t49 + t42 + t45 + t46*t49 - t46 + t48*t51)*np.exp(-rAB*t0*t1 - rCD*t4)
    if case_id == 91:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p**(-2.0)
        t7 = p + q
        t8 = F1*t7**(-1.5)/q**2
        t9 = t6*t8
        t10 = CDx*gamma
        t11 = CDy*delta*t10
        t12 = t11*t9
        t13 = F0*t11*t7**(-0.5)/q**3
        t14 = ABz**2
        t15 = 2*alpha
        t16 = beta*t14*t15
        t17 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t18 = t10*t17
        t19 = t0*t18
        t20 = F2*t7**(-2.5)
        t21 = t0*t20
        t22 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t23 = CDy*t22
        t24 = t23*t4
        t25 = t21*t24
        t26 = t20*t3
        t27 = t19*t26
        t28 = delta*t23
        t29 = t28*t8
        t30 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t31 = 2*t30**2
        t32 = ABz*t30
        t33 = 2*beta
        t34 = t32*t33
        t35 = t17*t22
        t36 = F3*t7**(-3.5)
        t37 = t35*t36
        t38 = t26*t35
        t39 = t31*t36
        t40 = 2*t1*t32
        t41 = t32*t37
        return np.pi**2.5*(-CDx*CDy*t21*t31*t5 + F4*p*q*t31*t35*t7**(-4.5) - t0*t29 + t12*t15*t32 - t12*t34 + t12 - t13*t6 - 2*t14*t2*t38 - t15*t41 - t16*t18*t9 + t16*t29*t6 - t18*t26*t40 + t18*t39 + t19*t8 + t20*t24*t40 - t25*t34 + t25 + t27*t34 - t27 - t28*t39 + t33*t41 - t37 + t38 + t13*t16/p**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 92:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = p**(-2.0)
        t7 = CDx*t6
        t8 = t3*t5*t7
        t9 = F0*delta*t4**(-0.5)/q**2
        t10 = 2*ABz**2*t1
        t11 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t12 = t0*t11
        t13 = F2*t4**(-2.5)
        t14 = t12*t13
        t15 = t2*t5
        t16 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t17 = 2*t16**2
        t18 = 2*ABz*t16
        t19 = t18*t8
        t20 = t14*t18
        return np.pi**2.5*(-CDx*delta*t0*t13*t17 + CDx*t10*t9/p**3 + F3*q*t11*t17*t4**(-3.5) + alpha*t19 - alpha*t20 - beta*t19 + beta*t20 - t10*t11*t15*t6 + t12*t15 - t14 - t7*t9 + t8)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 93:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/q**2
        t8 = p**(-2.0)
        t9 = CDx*delta
        t10 = t8*t9
        t11 = t10*t7
        t12 = ABz*t11
        t13 = CDz*gamma
        t14 = F0*t13*t6**(-0.5)/q**3
        t15 = 2*ABz**2
        t16 = alpha*t15
        t17 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t18 = t17**3
        t19 = F3*t6**(-3.5)
        t20 = 2*t19
        t21 = F2*t6**(-2.5)
        t22 = t21*t3
        t23 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t24 = ABz*t23
        t25 = t22*t24
        t26 = CDz*t23
        t27 = gamma*t26
        t28 = t0*t7
        t29 = beta*t0
        t30 = t17*t9
        t31 = t0*t21
        t32 = t17**2
        t33 = 2*t32
        t34 = t22*t33*t9
        t35 = ABz*t1
        t36 = ABz*t29
        t37 = 2*ABz*t13*t30
        t38 = beta*t7*t8
        t39 = t16*t38
        t40 = t17*t23
        t41 = t22*t40
        t42 = t20*t32
        t43 = t24*t42
        t44 = 2*CDz*t21*t4*t40
        return np.pi**2.5*(-CDx*CDz*t31*t33*t5 + 2*F4*p*q*t18*t23*t6**(-4.5) - alpha*t12 + alpha*t37*t7*t8 - alpha*t43 + beta*t12 + beta*t43 + beta*t14*t16*t9/p**3 + 3*t0*t22*t30 + t1*t25 - t10*t14 + t11*t13 - t15*t2*t41 - t18*t20*t9 - 3*t19*t40 - t25*t29 - t26*t31*t4 + t27*t28 - t27*t39 + t27*t42 - t28*t30 + t30*t39 + t34*t35 - t34*t36 - t35*t44 + t36*t44 - t37*t38 + t41)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 94:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p**(-2.0)
        t7 = p + q
        t8 = F1*t7**(-1.5)/q**2
        t9 = t6*t8
        t10 = CDy*gamma
        t11 = CDx*delta*t10
        t12 = t11*t9
        t13 = F0*t11*t7**(-0.5)/q**3
        t14 = ABz**2
        t15 = 2*alpha
        t16 = beta*t14*t15
        t17 = F2*t7**(-2.5)
        t18 = t0*t17
        t19 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t20 = CDx*t19
        t21 = t20*t4
        t22 = t18*t21
        t23 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t24 = t10*t23
        t25 = t0*t24
        t26 = delta*t20
        t27 = t26*t8
        t28 = t17*t3
        t29 = t25*t28
        t30 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t31 = 2*t30**2
        t32 = ABz*t30
        t33 = t15*t32
        t34 = 2*beta
        t35 = t19*t23
        t36 = F3*t7**(-3.5)
        t37 = t35*t36
        t38 = t28*t35
        t39 = t31*t36
        t40 = 2*t1*t32
        t41 = t32*t37
        return np.pi**2.5*(-CDx*CDy*t18*t31*t5 + F4*p*q*t31*t35*t7**(-4.5) - t0*t27 - t12*t32*t34 + t12*t33 + t12 - t13*t6 - 2*t14*t2*t38 - t15*t41 - t16*t24*t9 + t16*t27*t6 - t17*t21*t40 + t22*t33 + t22 + t24*t28*t40 + t24*t39 + t25*t8 - t26*t39 - t29*t33 - t29 + t34*t41 - t37 + t38 + t13*t16/p**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 95:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = gamma*t3
        t5 = q**(-2.0)
        t6 = (1/2)*t5
        t7 = p**(-2.0)
        t8 = p + q
        t9 = F0*t8**(-0.5)
        t10 = t7*t9
        t11 = (1/2)*t2
        t12 = F1*t8**(-1.5)
        t13 = t12*t7
        t14 = t0*t12
        t15 = F2*t8**(-2.5)
        t16 = t0*t15
        t17 = ABz**2*t1
        t18 = t17*t5
        t19 = t13*t18
        t20 = CDx**2
        t21 = delta*t20
        t22 = gamma*t5
        t23 = t13*t22
        t24 = t9/p**3
        t25 = gamma/q**3
        t26 = 2*delta
        t27 = t20*t26
        t28 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t29 = t28**2
        t30 = F3*t8**(-3.5)
        t31 = t29*t30
        t32 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t33 = t32**2
        t34 = t30*t33
        t35 = t16*t33
        t36 = t2*t29
        t37 = ABz*t32
        t38 = beta*t37
        t39 = t13*t2
        t40 = alpha*t37
        t41 = t16*t2
        t42 = t40*t41
        t43 = CDx*t28
        t44 = t14*t43
        t45 = t16*t3*t43
        t46 = t38*t41
        t47 = t23*t27
        t48 = 2*gamma
        t49 = t43*t48
        t50 = 2*t31
        t51 = t34*t43
        t52 = 2*t45
        return np.pi**2.5*(2*F4*p*q*t29*t33*t8**(-4.5) - delta*t44*t5 - gamma*t41*t43 - t10*t21*t25 + t10*t6 - t11*t13 + t11*t16 - t14*t6 + t15*t36 - 2*t16*t17*t36 + t17*t24*t25*t27 - t18*t24 + t19*t26*t43 - t19*t49 + t19 - 2*t20*t35*t4 + t21*t23 + t22*t44 - t26*t51 - t31 - t34 + t35 + t38*t39 - t38*t47 + t38*t50 - t38*t52 - t39*t40 + t40*t47 - t40*t50 + t40*t52 - t42*t49 + t42 + t45 + t46*t49 - t46 + t48*t51)*np.exp(-rAB*t0*t1 - rCD*t4)
    if case_id == 96:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = ABy*alpha
        t4 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        t5 = t2**(-1.5)
        t6 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        return 2*np.pi**2.5*t0*(-ABz*F0*beta*t1*t2**(-0.5)*t3/p**2 + ABz*F1*beta*t0*t5*t6 - F1*t0*t3*t4*t5 + F2*q*t2**(-2.5)*t4*t6)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 97:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = ABy*ABz*t1
        t8 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t9 = t5**(-2.5)
        t10 = F2*t0*t9
        t11 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t12 = t11**2
        t13 = ABy*alpha
        t14 = F1*t4*t6
        t15 = 2*t11
        return np.pi**2.5*(ABy*F1*alpha*t2*t4*t6 + 2*ABz*CDz*F1*beta*gamma*t2*t4*t6*t8 + 2*ABz*F2*beta*t0*t11*t8*t9 - 2*CDz*F0*gamma*t5**(-0.5)*t7/(p**3*q**2) + 2*CDz*F2*gamma*t0*t11*t8*t9 - CDz*t13*t14*t15*t3 + 2*F3*q*t12*t5**(-3.5)*t8 - 2*t10*t12*t13 - t10*t8 - t14*t15*t2*t7)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 98:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = F1*t4*t6
        t8 = t2*t7
        t9 = ABy*ABz*t1
        t10 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t11 = t5**(-2.5)
        t12 = F2*t0*t10*t11
        t13 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t14 = t13**2
        t15 = 2*t13
        t16 = ABy*alpha
        return np.pi**2.5*(2*ABz*CDy*F1*beta*gamma*t13*t2*t4*t6 + 2*ABz*F2*beta*t0*t11*t14 - ABz*beta*t8 - 2*CDy*F0*gamma*t5**(-0.5)*t9/(p**3*q**2) + 2*CDy*F2*gamma*t0*t10*t11*t13 - 2*CDy*t10*t16*t3*t7 + 2*F3*q*t10*t14*t5**(-3.5) - t12*t15*t16 - t12 - t15*t8*t9)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 99:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = gamma*t1
        t3 = p + q
        t4 = ABz*beta
        t5 = ABy*alpha
        t6 = CDx*t5
        t7 = p**(-2.0)
        t8 = t3**(-1.5)
        t9 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        t10 = t5*t9
        t11 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        t12 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        t13 = t3**(-2.5)
        return 2*np.pi**2.5*(ABz*CDx*F1*beta*gamma*t1*t12*t7*t8 + ABz*F2*beta*t0*t12*t13*t9 + CDx*F2*gamma*t0*t11*t12*t13 - F0*gamma*t3**(-0.5)*t4*t6/(p**3*q**2) - F1*t1*t10*t4*t7*t8 - F1*t11*t2*t6*t7*t8 - F2*t0*t10*t11*t13 + F3*q*t11*t12*t3**(-3.5)*t9)*np.exp(-alpha*beta*rAB*t0 - delta*rCD*t2)
    if case_id == 100:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = ABy*alpha
        t5 = p + q
        t6 = F1*t5**(-1.5)/p**2
        t7 = t2*t6
        t8 = CDz*delta
        t9 = ABy*ABz*t1
        t10 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t11 = F2*t0*t5**(-2.5)
        t12 = t10*t11
        t13 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t14 = 2*t13**2
        t15 = 2*t13
        t16 = CDz*t3*t6
        t17 = ABz*beta
        t18 = t12*t15
        return np.pi**2.5*(2*F0*t5**(-0.5)*t8*t9/(p**3*q**2) + F3*q*t10*t14*t5**(-3.5) - 2*t10*t16*t17 - t11*t14*t4 - t12 + t15*t16*t4 - t15*t7*t9 + t17*t18 - t18*t8 + t4*t7)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 101:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = ABy*alpha
        t7 = q**(-2.0)
        t8 = p + q
        t9 = F1*t8**(-1.5)/p**2
        t10 = t7*t9
        t11 = t10*t6
        t12 = ABz*beta
        t13 = CDz*t11
        t14 = F0*t12*t6*t8**(-0.5)/p**3
        t15 = 2*CDz**2
        t16 = delta*t15
        t17 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t18 = t17**3
        t19 = F3*t8**(-3.5)
        t20 = 2*t19
        t21 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t22 = ABz*t21
        t23 = beta*t22
        t24 = t3*t9
        t25 = F2*t8**(-2.5)
        t26 = t0*t25
        t27 = CDz*t21
        t28 = t26*t27
        t29 = t17*t6
        t30 = t25*t3
        t31 = t1*t30
        t32 = gamma*t3
        t33 = t17**2
        t34 = 2*t33
        t35 = CDz*t26*t34*t6
        t36 = 2*CDz
        t37 = t12*t29*t36
        t38 = gamma*t10
        t39 = t16*t38
        t40 = t17*t21
        t41 = t26*t40
        t42 = t20*t33
        t43 = t27*t42
        t44 = ABz*t36*t40
        return np.pi**2.5*(-ABy*ABz*t2*t30*t34 + 2*F4*p*q*t18*t21*t8**(-4.5) + delta*t10*t37 - delta*t13 - delta*t43 + gamma*t13 + gamma*t31*t44 + gamma*t43 + gamma*t14*t16/q**3 - t1*t25*t4*t44 + t11*t12 - t14*t7 - t15*t41*t5 - t18*t20*t6 - 3*t19*t40 - t22*t31 + t23*t24 - t23*t39 + t23*t42 - t24*t29 + 3*t26*t29*t3 - t28*t32 + t28*t4 + t29*t39 - t32*t35 + t35*t4 - t37*t38 + t41)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 102:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F2*t6**(-2.5)
        t8 = t3*t7
        t9 = t0*t8
        t10 = CDy*gamma
        t11 = ABy*t10
        t12 = alpha*t11
        t13 = F1*t6**(-1.5)/(p**2*q**2)
        t14 = t12*t13
        t15 = ABz*beta
        t16 = CDz*t15
        t17 = delta*t16
        t18 = t13*t17
        t19 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t20 = t19**2
        t21 = F3*t6**(-3.5)
        t22 = t20*t21
        t23 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t24 = t23**2
        t25 = t21*t24
        t26 = ABy*t19
        t27 = t1*t8
        t28 = CDz*t23
        t29 = t0*t4*t7
        t30 = t15*t23
        t31 = t30*t9
        t32 = t10*t19
        t33 = 2*t24
        t34 = 2*t14
        t35 = alpha*t26
        t36 = delta*t28
        t37 = 2*t32
        t38 = 2*t25
        t39 = 2*t22
        t40 = 2*t26
        t41 = t28*t7
        return np.pi**2.5*(-ABz*t2*t23*t40*t8 - 2*CDy*t0*t19*t41*t5 + 2*F0*t12*t17*t6**(-0.5)/(p**3*q**3) + F4*p*q*t20*t33*t6**(-4.5) + t1*t4*t40*t41 - t11*t27*t33 + t14 - 2*t16*t20*t29 + 2*t18*t35 - t18*t37 + t18 - t22 - t25 + t26*t27 + t28*t29 - t30*t34 + t30*t39 + t31*t37 - t31 + t32*t38 - t32*t9 + t34*t36 - t35*t38 - t36*t39 + (1/2)*t9)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 103:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABy*CDx
        t9 = alpha*gamma*t8
        t10 = t7*t9
        t11 = CDz*delta
        t12 = ABz*beta
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t1*t14
        t16 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t17 = ABy*t16
        t18 = t17*t3
        t19 = t15*t18
        t20 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t21 = CDx*t20
        t22 = t0*t14
        t23 = t21*t22
        t24 = t23*t4
        t25 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t26 = 2*t25**2
        t27 = 2*t25
        t28 = t12*t27
        t29 = alpha*t17
        t30 = t13*t7
        t31 = t11*t27
        t32 = gamma*t21
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t27*t35
        return np.pi**2.5*(-ABz*t14*t18*t2*t27 - CDz*t23*t27*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) - t10*t28 + t10*t31 + t10 - t11*t37 + t12*t37 - t13*t22*t3*t33 - t15*t26*t4*t8 + t19*t31 + t19 + t24*t28 - t24 + t29*t30 - t29*t36 - t30*t32 + t32*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 104:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = ABz*beta
        t5 = p**(-2.0)
        t6 = p + q
        t7 = t6**(-1.5)
        t8 = F1*t5*t7
        t9 = t2*t8
        t10 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t11 = t6**(-2.5)
        t12 = F2*t0*t10*t11
        t13 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t14 = t13**2
        t15 = 2*t13
        t16 = t12*t15
        return np.pi**2.5*(2*ABy*ABz*CDy*F0*alpha*beta*delta*t6**(-0.5)/(p**3*q**2) - ABy*ABz*t1*t15*t9 + 2*ABy*CDy*F1*alpha*delta*t10*t2*t5*t7 - ABy*alpha*t16 + 2*ABz*F2*beta*t0*t11*t14 - CDy*delta*t16 - CDy*t15*t3*t4*t8 + 2*F3*q*t10*t14*t6**(-3.5) - t12 - t4*t9)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 105:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = t6**(-2.5)
        t8 = p**(-2.0)
        t9 = q**(-2.0)
        t10 = t6**(-1.5)
        t11 = F1*t10*t8*t9
        t12 = ABz*beta
        t13 = CDz*gamma
        t14 = t11*t12*t13
        t15 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t16 = t15**2
        t17 = t6**(-3.5)
        t18 = F3*t17
        t19 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t20 = t19**2
        t21 = t18*t20
        t22 = F2*t7
        t23 = t22*t3
        t24 = t0*t19*t23
        t25 = ABy*t15
        t26 = alpha*t25
        t27 = 2*t14
        t28 = CDy*t15
        t29 = delta*t28
        t30 = 2*t21
        t31 = 2*t19
        t32 = t23*t25
        t33 = t0*t28
        return np.pi**2.5*(2*ABy*ABz*CDy*CDz*F0*alpha*beta*delta*gamma*t6**(-0.5)/(p**3*q**3) + 2*ABy*ABz*CDy*F1*alpha*beta*delta*t10*t19*t8*t9 + 2*ABy*CDy*CDz*F1*alpha*delta*gamma*t10*t19*t8*t9 + 2*ABy*CDy*F2*alpha*delta*t0*t20*t3*t7 - ABy*CDy*alpha*delta*t11 + ABy*F2*alpha*t0*t15*t3*t7 + 2*ABz*CDz*F2*beta*gamma*t0*t16*t3*t7 + 2*ABz*F3*beta*t16*t17*t19 - ABz*t2*t31*t32 + CDy*F2*delta*t0*t15*t3*t7 + 2*CDz*F3*gamma*t16*t17*t19 - CDz*t22*t31*t33*t5 + (1/2)*F2*t0*t3*t7 + 2*F4*p*q*t16*t20*t6**(-4.5) - t1*t13*t31*t32 - t12*t22*t31*t33*t4 - t12*t24 - t13*t24 - t14 - t16*t18 - t21 - t26*t27 - t26*t30 - t27*t29 - t29*t30)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 106:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = q**(-2.0)
        t7 = ABz*beta
        t8 = p + q
        t9 = F1*t8**(-1.5)/p**2
        t10 = t6*t7*t9
        t11 = ABy*alpha
        t12 = t10*t11
        t13 = CDy*t10
        t14 = F0*t11*t7*t8**(-0.5)/p**3
        t15 = 2*CDy**2
        t16 = gamma*t15
        t17 = delta*t16
        t18 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t19 = t18**3
        t20 = F3*t8**(-3.5)
        t21 = 2*t20
        t22 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t23 = ABy*t22
        t24 = F2*t8**(-2.5)
        t25 = t24*t3
        t26 = t1*t25
        t27 = t3*t7
        t28 = t18*t27
        t29 = t0*t24
        t30 = CDy*t22
        t31 = t29*t30
        t32 = alpha*t23
        t33 = t32*t9
        t34 = t18**2
        t35 = 2*t34
        t36 = CDy*t4
        t37 = t29*t35
        t38 = CDy*gamma
        t39 = delta*t18
        t40 = 2*t38
        t41 = t18*t22
        t42 = t29*t41
        t43 = t21*t34
        t44 = t30*t43
        t45 = ABy*t41
        return np.pi**2.5*(-ABy*ABz*t2*t25*t35 + 2*CDy*t12*t39 + 2*F4*p*q*t19*t22*t8**(-4.5) + delta*t13 - delta*t44 - gamma*t13 - gamma*t3*t31 + gamma*t44 + 2*t1*t24*t36*t45 - t10*t16*t39 - t12*t18*t40 + t12 - t14*t6 - t15*t42*t5 + t17*t33*t6 + t19*t21*t7 - 3*t20*t41 + t23*t26 - t26*t40*t45 + t27*t37*t38 - 3*t28*t29 + t28*t9 - t3*t33 + t31*t4 - t32*t43 - t36*t37*t7 + t42 + t14*t17/q**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 107:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p**(-2.0)
        t7 = q**(-2.0)
        t8 = p + q
        t9 = t8**(-1.5)
        t10 = ABz*CDx*F1*beta*gamma*t6*t7*t9
        t11 = t8**(-2.5)
        t12 = F2*t11
        t13 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t14 = ABz*t13*t3
        t15 = t1*t12*t14
        t16 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t17 = CDx*t0*t12*t16
        t18 = t17*t4
        t19 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t20 = t19**2
        t21 = 2*t19
        t22 = ABy*alpha
        t23 = t21*t22
        t24 = CDy*delta
        t25 = t21*t24
        t26 = t8**(-3.5)
        t27 = F3*t13*t16*t26
        t28 = t21*t27
        return np.pi**2.5*(2*ABy*ABz*CDx*CDy*F0*alpha*beta*delta*gamma*t8**(-0.5)/(p**3*q**3) + 2*ABy*ABz*CDy*F1*alpha*beta*delta*t13*t6*t7*t9 + 2*ABy*CDx*CDy*F1*alpha*delta*gamma*t16*t6*t7*t9 + 2*ABy*CDy*F2*alpha*delta*t0*t11*t13*t16*t3 - ABy*t12*t14*t2*t21 + 2*ABz*CDx*F2*beta*gamma*t0*t11*t20*t3 + 2*ABz*F3*beta*t13*t20*t26 + 2*CDx*F3*gamma*t16*t20*t26 - CDy*t17*t21*t5 + 2*F4*p*q*t13*t16*t20*t8**(-4.5) - t10*t23 - t10*t25 - t10 - t15*t25 - t15 - t18*t23 - t18 - t22*t28 - t24*t28 - t27)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 108:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = ABy*alpha
        t6 = CDx*t5
        t7 = ABz*beta
        t8 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t9 = F1*t4**(-1.5)/p**2
        t10 = t3*t9
        t11 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t12 = t11*t5
        t13 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t14 = CDx*t13
        t15 = t11*t13
        t16 = F2*t4**(-2.5)
        t17 = t0*t16*t8
        return 2*np.pi**2.5*(ABz*t1*t15*t16 + F0*delta*t4**(-0.5)*t6*t7/(p**3*q**2) + F3*q*t15*t4**(-3.5)*t8 - delta*t14*t17 - t10*t14*t7 + t10*t6*t8 - t12*t17 - t12*t2*t7*t9)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 109:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABy*CDx
        t9 = alpha*delta*t8
        t10 = t7*t9
        t11 = ABz*beta
        t12 = CDz*gamma
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t1*t14
        t16 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t17 = ABy*t16
        t18 = t17*t3
        t19 = t15*t18
        t20 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t21 = CDx*t20
        t22 = t0*t14
        t23 = t21*t22
        t24 = t23*t4
        t25 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t26 = 2*t25**2
        t27 = 2*t25
        t28 = t11*t27
        t29 = alpha*t17
        t30 = t13*t7
        t31 = t12*t27
        t32 = delta*t21
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t27*t35
        return np.pi**2.5*(-ABz*t14*t18*t2*t27 - CDz*t23*t27*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) + t10*t28 + t10*t31 - t10 + t11*t37 + t12*t37 + t13*t22*t3*t33 + t15*t26*t4*t8 - t19*t31 + t19 - t24*t28 + t24 - t29*t30 - t29*t36 - t30*t32 - t32*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 110:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABz*CDx
        t9 = beta*delta*t8
        t10 = t7*t9
        t11 = ABy*alpha
        t12 = CDy*gamma
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t14*t4
        t16 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t17 = CDx*t16
        t18 = t0*t17
        t19 = t15*t18
        t20 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t21 = ABz*t20
        t22 = t14*t3
        t23 = t21*t22
        t24 = t1*t23
        t25 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t26 = 2*t25**2
        t27 = 2*t25
        t28 = t11*t27
        t29 = beta*t21
        t30 = t13*t7
        t31 = delta*t17
        t32 = t12*t27
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t27*t35
        return np.pi**2.5*(-ABy*t2*t23*t27 - CDy*t14*t18*t27*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) - t0*t13*t22*t33 - t1*t15*t26*t8 + t10*t28 - t10*t32 + t10 - t11*t37 + t12*t37 + t19*t28 + t19 + t24*t32 - t24 - t29*t30 + t29*t36 + t30*t31 - t31*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 111:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = q**(-2.0)
        t7 = p + q
        t8 = F1*t7**(-1.5)/p**2
        t9 = t6*t8
        t10 = ABz*beta
        t11 = ABy*alpha*t10
        t12 = t11*t9
        t13 = F0*t11*t7**(-0.5)/p**3
        t14 = CDx**2
        t15 = 2*delta
        t16 = gamma*t14*t15
        t17 = F2*t7**(-2.5)
        t18 = t17*t3
        t19 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t20 = ABy*t19
        t21 = t1*t20
        t22 = t18*t21
        t23 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t24 = t10*t23
        t25 = t24*t3
        t26 = alpha*t20
        t27 = t26*t8
        t28 = t0*t17
        t29 = t25*t28
        t30 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t31 = 2*t30**2
        t32 = CDx*t30
        t33 = t15*t32
        t34 = 2*gamma
        t35 = t19*t23
        t36 = F3*t7**(-3.5)
        t37 = t35*t36
        t38 = t28*t35
        t39 = t31*t36
        t40 = 2*t32*t4
        t41 = t32*t37
        return np.pi**2.5*(-ABy*ABz*t18*t2*t31 + F4*p*q*t31*t35*t7**(-4.5) - t12*t32*t34 + t12*t33 + t12 - t13*t6 - 2*t14*t38*t5 - t15*t41 - t16*t24*t9 + t16*t27*t6 - t17*t21*t40 + t22*t33 + t22 + t24*t28*t40 + t24*t39 + t25*t8 - t26*t39 - t27*t3 - t29*t33 - t29 + t34*t41 - t37 + t38 + t13*t16/q**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 112:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = ABx*alpha
        t4 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        t5 = t2**(-1.5)
        t6 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        return 2*np.pi**2.5*t0*(-ABz*F0*beta*t1*t2**(-0.5)*t3/p**2 + ABz*F1*beta*t0*t5*t6 - F1*t0*t3*t4*t5 + F2*q*t2**(-2.5)*t4*t6)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 113:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = ABx*ABz*t1
        t8 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t9 = t5**(-2.5)
        t10 = F2*t0*t9
        t11 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t12 = t11**2
        t13 = ABx*alpha
        t14 = F1*t4*t6
        t15 = 2*t11
        return np.pi**2.5*(ABx*F1*alpha*t2*t4*t6 + 2*ABz*CDz*F1*beta*gamma*t2*t4*t6*t8 + 2*ABz*F2*beta*t0*t11*t8*t9 - 2*CDz*F0*gamma*t5**(-0.5)*t7/(p**3*q**2) + 2*CDz*F2*gamma*t0*t11*t8*t9 - CDz*t13*t14*t15*t3 + 2*F3*q*t12*t5**(-3.5)*t8 - 2*t10*t12*t13 - t10*t8 - t14*t15*t2*t7)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 114:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = gamma*t1
        t3 = p + q
        t4 = ABz*beta
        t5 = ABx*alpha
        t6 = CDy*t5
        t7 = p**(-2.0)
        t8 = t3**(-1.5)
        t9 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        t10 = t5*t9
        t11 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        t12 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        t13 = t3**(-2.5)
        return 2*np.pi**2.5*(ABz*CDy*F1*beta*gamma*t1*t12*t7*t8 + ABz*F2*beta*t0*t12*t13*t9 + CDy*F2*gamma*t0*t11*t12*t13 - F0*gamma*t3**(-0.5)*t4*t6/(p**3*q**2) - F1*t1*t10*t4*t7*t8 - F1*t11*t2*t6*t7*t8 - F2*t0*t10*t11*t13 + F3*q*t11*t12*t3**(-3.5)*t9)*np.exp(-alpha*beta*rAB*t0 - delta*rCD*t2)
    if case_id == 115:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = F1*t4*t6
        t8 = t2*t7
        t9 = ABx*ABz*t1
        t10 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t11 = t5**(-2.5)
        t12 = F2*t0*t10*t11
        t13 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t14 = t13**2
        t15 = 2*t13
        t16 = ABx*alpha
        return np.pi**2.5*(2*ABz*CDx*F1*beta*gamma*t13*t2*t4*t6 + 2*ABz*F2*beta*t0*t11*t14 - ABz*beta*t8 - 2*CDx*F0*gamma*t5**(-0.5)*t9/(p**3*q**2) + 2*CDx*F2*gamma*t0*t10*t11*t13 - 2*CDx*t10*t16*t3*t7 + 2*F3*q*t10*t14*t5**(-3.5) - t12*t15*t16 - t12 - t15*t8*t9)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 116:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = ABx*alpha
        t5 = p + q
        t6 = F1*t5**(-1.5)/p**2
        t7 = t2*t6
        t8 = CDz*delta
        t9 = ABx*ABz*t1
        t10 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t11 = F2*t0*t5**(-2.5)
        t12 = t10*t11
        t13 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t14 = 2*t13**2
        t15 = 2*t13
        t16 = CDz*t3*t6
        t17 = ABz*beta
        t18 = t12*t15
        return np.pi**2.5*(2*F0*t5**(-0.5)*t8*t9/(p**3*q**2) + F3*q*t10*t14*t5**(-3.5) - 2*t10*t16*t17 - t11*t14*t4 - t12 + t15*t16*t4 - t15*t7*t9 + t17*t18 - t18*t8 + t4*t7)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 117:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = ABx*alpha
        t7 = q**(-2.0)
        t8 = p + q
        t9 = F1*t8**(-1.5)/p**2
        t10 = t7*t9
        t11 = t10*t6
        t12 = ABz*beta
        t13 = CDz*t11
        t14 = F0*t12*t6*t8**(-0.5)/p**3
        t15 = 2*CDz**2
        t16 = delta*t15
        t17 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t18 = t17**3
        t19 = F3*t8**(-3.5)
        t20 = 2*t19
        t21 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t22 = ABz*t21
        t23 = beta*t22
        t24 = t3*t9
        t25 = F2*t8**(-2.5)
        t26 = t0*t25
        t27 = CDz*t21
        t28 = t26*t27
        t29 = t17*t6
        t30 = t25*t3
        t31 = t1*t30
        t32 = gamma*t3
        t33 = t17**2
        t34 = 2*t33
        t35 = CDz*t26*t34*t6
        t36 = 2*CDz
        t37 = t12*t29*t36
        t38 = gamma*t10
        t39 = t16*t38
        t40 = t17*t21
        t41 = t26*t40
        t42 = t20*t33
        t43 = t27*t42
        t44 = ABz*t36*t40
        return np.pi**2.5*(-ABx*ABz*t2*t30*t34 + 2*F4*p*q*t18*t21*t8**(-4.5) + delta*t10*t37 - delta*t13 - delta*t43 + gamma*t13 + gamma*t31*t44 + gamma*t43 + gamma*t14*t16/q**3 - t1*t25*t4*t44 + t11*t12 - t14*t7 - t15*t41*t5 - t18*t20*t6 - 3*t19*t40 - t22*t31 + t23*t24 - t23*t39 + t23*t42 - t24*t29 + 3*t26*t29*t3 - t28*t32 + t28*t4 + t29*t39 - t32*t35 + t35*t4 - t37*t38 + t41)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 118:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABx*CDy
        t9 = alpha*gamma*t8
        t10 = t7*t9
        t11 = CDz*delta
        t12 = ABz*beta
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t1*t14
        t16 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t17 = ABx*t16
        t18 = t17*t3
        t19 = t15*t18
        t20 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t21 = CDy*t20
        t22 = t0*t14
        t23 = t21*t22
        t24 = t23*t4
        t25 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t26 = 2*t25**2
        t27 = 2*t25
        t28 = t12*t27
        t29 = alpha*t17
        t30 = t13*t7
        t31 = t11*t27
        t32 = gamma*t21
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t27*t35
        return np.pi**2.5*(-ABz*t14*t18*t2*t27 - CDz*t23*t27*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) - t10*t28 + t10*t31 + t10 - t11*t37 + t12*t37 - t13*t22*t3*t33 - t15*t26*t4*t8 + t19*t31 + t19 + t24*t28 - t24 + t29*t30 - t29*t36 - t30*t32 + t32*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 119:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F2*t6**(-2.5)
        t8 = t3*t7
        t9 = t0*t8
        t10 = CDx*gamma
        t11 = ABx*t10
        t12 = alpha*t11
        t13 = F1*t6**(-1.5)/(p**2*q**2)
        t14 = t12*t13
        t15 = ABz*beta
        t16 = CDz*t15
        t17 = delta*t16
        t18 = t13*t17
        t19 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t20 = t19**2
        t21 = F3*t6**(-3.5)
        t22 = t20*t21
        t23 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t24 = t23**2
        t25 = t21*t24
        t26 = ABx*t19
        t27 = t1*t8
        t28 = CDz*t23
        t29 = t0*t4*t7
        t30 = t15*t23
        t31 = t30*t9
        t32 = t10*t19
        t33 = 2*t24
        t34 = 2*t14
        t35 = alpha*t26
        t36 = delta*t28
        t37 = 2*t32
        t38 = 2*t25
        t39 = 2*t22
        t40 = 2*t26
        t41 = t28*t7
        return np.pi**2.5*(-ABz*t2*t23*t40*t8 - 2*CDx*t0*t19*t41*t5 + 2*F0*t12*t17*t6**(-0.5)/(p**3*q**3) + F4*p*q*t20*t33*t6**(-4.5) + t1*t4*t40*t41 - t11*t27*t33 + t14 - 2*t16*t20*t29 + 2*t18*t35 - t18*t37 + t18 - t22 - t25 + t26*t27 + t28*t29 - t30*t34 + t30*t39 + t31*t37 - t31 + t32*t38 - t32*t9 + t34*t36 - t35*t38 - t36*t39 + (1/2)*t9)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 120:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = ABx*alpha
        t6 = CDy*t5
        t7 = ABz*beta
        t8 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t9 = F1*t4**(-1.5)/p**2
        t10 = t3*t9
        t11 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t12 = t11*t5
        t13 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t14 = CDy*t13
        t15 = t11*t13
        t16 = F2*t4**(-2.5)
        t17 = t0*t16*t8
        return 2*np.pi**2.5*(ABz*t1*t15*t16 + F0*delta*t4**(-0.5)*t6*t7/(p**3*q**2) + F3*q*t15*t4**(-3.5)*t8 - delta*t14*t17 - t10*t14*t7 + t10*t6*t8 - t12*t17 - t12*t2*t7*t9)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 121:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABx*CDy
        t9 = alpha*delta*t8
        t10 = t7*t9
        t11 = ABz*beta
        t12 = CDz*gamma
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t1*t14
        t16 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t17 = ABx*t16
        t18 = t17*t3
        t19 = t15*t18
        t20 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t21 = CDy*t20
        t22 = t0*t14
        t23 = t21*t22
        t24 = t23*t4
        t25 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t26 = 2*t25**2
        t27 = 2*t25
        t28 = t11*t27
        t29 = alpha*t17
        t30 = t13*t7
        t31 = t12*t27
        t32 = delta*t21
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t27*t35
        return np.pi**2.5*(-ABz*t14*t18*t2*t27 - CDz*t23*t27*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) + t10*t28 + t10*t31 - t10 + t11*t37 + t12*t37 + t13*t22*t3*t33 + t15*t26*t4*t8 - t19*t31 + t19 - t24*t28 + t24 - t29*t30 - t29*t36 - t30*t32 - t32*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 122:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = q**(-2.0)
        t7 = p + q
        t8 = F1*t7**(-1.5)/p**2
        t9 = t6*t8
        t10 = ABz*beta
        t11 = ABx*alpha*t10
        t12 = t11*t9
        t13 = F0*t11*t7**(-0.5)/p**3
        t14 = CDy**2
        t15 = 2*delta
        t16 = gamma*t14*t15
        t17 = F2*t7**(-2.5)
        t18 = t17*t3
        t19 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t20 = ABx*t19
        t21 = t1*t20
        t22 = t18*t21
        t23 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t24 = t10*t23
        t25 = t24*t3
        t26 = alpha*t20
        t27 = t26*t8
        t28 = t0*t17
        t29 = t25*t28
        t30 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t31 = 2*t30**2
        t32 = CDy*t30
        t33 = t15*t32
        t34 = 2*gamma
        t35 = t19*t23
        t36 = F3*t7**(-3.5)
        t37 = t35*t36
        t38 = t28*t35
        t39 = t31*t36
        t40 = 2*t32*t4
        t41 = t32*t37
        return np.pi**2.5*(-ABx*ABz*t18*t2*t31 + F4*p*q*t31*t35*t7**(-4.5) - t12*t32*t34 + t12*t33 + t12 - t13*t6 - 2*t14*t38*t5 - t15*t41 - t16*t24*t9 + t16*t27*t6 - t17*t21*t40 + t22*t33 + t22 + t24*t28*t40 + t24*t39 + t25*t8 - t26*t39 - t27*t3 - t29*t33 - t29 + t34*t41 - t37 + t38 + t13*t16/q**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 123:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABz*CDy
        t9 = beta*delta*t8
        t10 = t7*t9
        t11 = ABx*alpha
        t12 = CDx*gamma
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t14*t4
        t16 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t17 = CDy*t16
        t18 = t0*t17
        t19 = t15*t18
        t20 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t21 = ABz*t20
        t22 = t14*t3
        t23 = t21*t22
        t24 = t1*t23
        t25 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t26 = 2*t25**2
        t27 = beta*t21
        t28 = t13*t7
        t29 = 2*t25
        t30 = t11*t29
        t31 = delta*t17
        t32 = t12*t29
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t29*t35
        return np.pi**2.5*(-ABx*t2*t23*t29 - CDx*t14*t18*t29*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) - t0*t13*t22*t33 - t1*t15*t26*t8 + t10*t30 - t10*t32 + t10 - t11*t37 + t12*t37 + t19*t30 + t19 + t24*t32 - t24 - t27*t28 + t27*t36 + t28*t31 - t31*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 124:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = ABz*beta
        t5 = p**(-2.0)
        t6 = p + q
        t7 = t6**(-1.5)
        t8 = F1*t5*t7
        t9 = t2*t8
        t10 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t11 = t6**(-2.5)
        t12 = F2*t0*t10*t11
        t13 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t14 = t13**2
        t15 = 2*t13
        t16 = t12*t15
        return np.pi**2.5*(2*ABx*ABz*CDx*F0*alpha*beta*delta*t6**(-0.5)/(p**3*q**2) - ABx*ABz*t1*t15*t9 + 2*ABx*CDx*F1*alpha*delta*t10*t2*t5*t7 - ABx*alpha*t16 + 2*ABz*F2*beta*t0*t11*t14 - CDx*delta*t16 - CDx*t15*t3*t4*t8 + 2*F3*q*t10*t14*t6**(-3.5) - t12 - t4*t9)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 125:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = t6**(-2.5)
        t8 = p**(-2.0)
        t9 = q**(-2.0)
        t10 = t6**(-1.5)
        t11 = F1*t10*t8*t9
        t12 = ABz*beta
        t13 = CDz*gamma
        t14 = t11*t12*t13
        t15 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t16 = t15**2
        t17 = t6**(-3.5)
        t18 = F3*t17
        t19 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t20 = t19**2
        t21 = t18*t20
        t22 = F2*t7
        t23 = t22*t3
        t24 = t0*t19*t23
        t25 = ABx*t15
        t26 = alpha*t25
        t27 = 2*t14
        t28 = CDx*t15
        t29 = delta*t28
        t30 = 2*t21
        t31 = 2*t19
        t32 = t23*t25
        t33 = t0*t28
        return np.pi**2.5*(2*ABx*ABz*CDx*CDz*F0*alpha*beta*delta*gamma*t6**(-0.5)/(p**3*q**3) + 2*ABx*ABz*CDx*F1*alpha*beta*delta*t10*t19*t8*t9 + 2*ABx*CDx*CDz*F1*alpha*delta*gamma*t10*t19*t8*t9 + 2*ABx*CDx*F2*alpha*delta*t0*t20*t3*t7 - ABx*CDx*alpha*delta*t11 + ABx*F2*alpha*t0*t15*t3*t7 + 2*ABz*CDz*F2*beta*gamma*t0*t16*t3*t7 + 2*ABz*F3*beta*t16*t17*t19 - ABz*t2*t31*t32 + CDx*F2*delta*t0*t15*t3*t7 + 2*CDz*F3*gamma*t16*t17*t19 - CDz*t22*t31*t33*t5 + (1/2)*F2*t0*t3*t7 + 2*F4*p*q*t16*t20*t6**(-4.5) - t1*t13*t31*t32 - t12*t22*t31*t33*t4 - t12*t24 - t13*t24 - t14 - t16*t18 - t21 - t26*t27 - t26*t30 - t27*t29 - t29*t30)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 126:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p**(-2.0)
        t7 = q**(-2.0)
        t8 = p + q
        t9 = t8**(-1.5)
        t10 = ABz*CDy*F1*beta*gamma*t6*t7*t9
        t11 = t8**(-2.5)
        t12 = F2*t11
        t13 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t14 = ABz*t13*t3
        t15 = t1*t12*t14
        t16 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t17 = CDy*t0*t12*t16
        t18 = t17*t4
        t19 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t20 = t19**2
        t21 = 2*t19
        t22 = ABx*alpha
        t23 = t21*t22
        t24 = CDx*delta
        t25 = t21*t24
        t26 = t8**(-3.5)
        t27 = F3*t13*t16*t26
        t28 = t21*t27
        return np.pi**2.5*(2*ABx*ABz*CDx*CDy*F0*alpha*beta*delta*gamma*t8**(-0.5)/(p**3*q**3) + 2*ABx*ABz*CDx*F1*alpha*beta*delta*t13*t6*t7*t9 + 2*ABx*CDx*CDy*F1*alpha*delta*gamma*t16*t6*t7*t9 + 2*ABx*CDx*F2*alpha*delta*t0*t11*t13*t16*t3 - ABx*t12*t14*t2*t21 + 2*ABz*CDy*F2*beta*gamma*t0*t11*t20*t3 + 2*ABz*F3*beta*t13*t20*t26 - CDx*t17*t21*t5 + 2*CDy*F3*gamma*t16*t20*t26 + 2*F4*p*q*t13*t16*t20*t8**(-4.5) - t10*t23 - t10*t25 - t10 - t15*t25 - t15 - t18*t23 - t18 - t22*t28 - t24*t28 - t27)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 127:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = q**(-2.0)
        t7 = ABz*beta
        t8 = p + q
        t9 = F1*t8**(-1.5)/p**2
        t10 = t6*t7*t9
        t11 = ABx*alpha
        t12 = t10*t11
        t13 = CDx*t10
        t14 = F0*t11*t7*t8**(-0.5)/p**3
        t15 = 2*CDx**2
        t16 = gamma*t15
        t17 = delta*t16
        t18 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t19 = t18**3
        t20 = F3*t8**(-3.5)
        t21 = 2*t20
        t22 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t23 = ABx*t22
        t24 = F2*t8**(-2.5)
        t25 = t24*t3
        t26 = t1*t25
        t27 = t3*t7
        t28 = t18*t27
        t29 = t0*t24
        t30 = CDx*t22
        t31 = t29*t30
        t32 = alpha*t23
        t33 = t32*t9
        t34 = t18**2
        t35 = 2*t34
        t36 = CDx*t4
        t37 = t29*t35
        t38 = CDx*gamma
        t39 = delta*t18
        t40 = 2*t38
        t41 = t18*t22
        t42 = t29*t41
        t43 = t21*t34
        t44 = t30*t43
        t45 = ABx*t41
        return np.pi**2.5*(-ABx*ABz*t2*t25*t35 + 2*CDx*t12*t39 + 2*F4*p*q*t19*t22*t8**(-4.5) + delta*t13 - delta*t44 - gamma*t13 - gamma*t3*t31 + gamma*t44 + 2*t1*t24*t36*t45 - t10*t16*t39 - t12*t18*t40 + t12 - t14*t6 - t15*t42*t5 + t17*t33*t6 + t19*t21*t7 - 3*t20*t41 + t23*t26 - t26*t40*t45 + t27*t37*t38 - 3*t28*t29 + t28*t9 - t3*t33 + t31*t4 - t32*t43 - t36*t37*t7 + t42 + t14*t17/q**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 128:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = p + q
        return -2*np.pi**2.5*t0*(ABy*F0*t1*t2*t3**(-0.5) + F1*t3**(-1.5)*(t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)))*np.exp(-alpha*rAB*t1 - delta*gamma*rCD*t2)
    if case_id == 129:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        return -2*np.pi**2.5*(ABy*CDz*F0*beta*gamma*t4**(-0.5)/(p**2*q**2) + ABy*t1*t2*t5*t6 + CDz*t0*t3*t6*t7 + F2*t4**(-2.5)*t5*t7)*np.exp(-alpha*rAB*t1 - delta*rCD*t3)
    if case_id == 130:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = t2**(-1.5)
        t4 = ABy*beta
        t5 = CDy*gamma
        t6 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        t7 = 2*F1*t0*t1*t3*t6
        return np.pi**2.5*(-2*F0*t2**(-0.5)*t4*t5/(p**2*q**2) + F1*t0*t1*t3 - 2*F2*t2**(-2.5)*t6**2 - t4*t7 - t5*t7)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 131:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        return -2*np.pi**2.5*(ABy*CDx*F0*beta*gamma*t4**(-0.5)/(p**2*q**2) + ABy*t1*t2*t5*t6 + CDx*t0*t3*t6*t7 + F2*t4**(-2.5)*t5*t7)*np.exp(-alpha*rAB*t1 - delta*rCD*t3)
    if case_id == 132:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        return 2*np.pi**2.5*(ABy*CDz*F0*beta*delta*t4**(-0.5)/(p**2*q**2) - ABy*t1*t2*t6*t7 + CDz*t0*t3*t5*t6 - F2*t4**(-2.5)*t5*t7)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 133:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = q**(-2.0)
        t7 = ABy*t6
        t8 = t1*t5*t7
        t9 = F0*beta*t4**(-0.5)/p**2
        t10 = 2*CDz**2*t3
        t11 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t12 = t11*t2
        t13 = F2*t4**(-2.5)
        t14 = t12*t13
        t15 = t0*t5
        t16 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t17 = 2*t16**2
        t18 = 2*CDz*t16
        t19 = t18*t8
        t20 = t14*t18
        return np.pi**2.5*(-ABy*beta*t13*t17*t2 + ABy*t10*t9/q**3 - F3*p*t11*t17*t4**(-3.5) + delta*t19 + delta*t20 - gamma*t19 - gamma*t20 + t10*t11*t15*t6 - t12*t15 + t14 - t7*t9 + t8)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 134:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDz*delta
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABy
        t9 = beta*t8
        t10 = CDy*CDz*t3
        t11 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t15 = 2*t14**2
        t16 = CDy*gamma
        t17 = t1*t8
        t18 = t13*t14
        return np.pi**2.5*(F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 - t0*t7 - t11*t16*t17*t6 + t12*t15*t4 + t13 + t14*t17*t7 - 2*t16*t18 - t18*t9)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 135:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = CDz*delta
        t6 = CDx*gamma
        t7 = ABy*t6
        t8 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t9 = ABy*t8
        t10 = q**(-2.0)
        t11 = t4**(-1.5)
        t12 = F1*t1*t10*t11
        t13 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t14 = t13*t6
        t15 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t16 = t13*t8
        t17 = F2*t4**(-2.5)
        t18 = t15*t17*t2
        return 2*np.pi**2.5*(CDz*t16*t17*t3 + F0*beta*t4**(-0.5)*t5*t7/(p**2*q**3) + F1*t0*t10*t11*t14*t5 - F3*p*t15*t16*t4**(-3.5) - beta*t18*t9 - t12*t15*t7 + t12*t5*t9 - t14*t18)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 136:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = F1*t0*t1*t2**(-1.5)
        t4 = ABy*beta
        t5 = CDy*delta
        t6 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        t7 = 2*t3*t6
        return np.pi**2.5*(2*F0*t2**(-0.5)*t4*t5/(p**2*q**2) - 2*F2*t2**(-2.5)*t6**2 + t3 - t4*t7 + t5*t7)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 137:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDz*gamma
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABy
        t9 = beta*t8
        t10 = CDy*CDz*t3
        t11 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t15 = 2*t14**2
        t16 = CDy*delta
        t17 = t1*t8
        t18 = t13*t14
        return np.pi**2.5*(F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 + t0*t7 + t11*t16*t17*t6 - t12*t15*t4 + t13 - t14*t17*t7 + 2*t16*t18 - t18*t9)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 138:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = q**(-2.0)
        t7 = ABy*t6
        t8 = t1*t5*t7
        t9 = CDy*gamma
        t10 = t0*t5
        t11 = t10*t6
        t12 = F0*beta*t4**(-0.5)/p**2
        t13 = CDy*delta
        t14 = 2*ABy
        t15 = CDy**2*delta*gamma
        t16 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t17 = t16*t2
        t18 = F2*t4**(-2.5)
        t19 = t16**2*t18
        t20 = t19*t2
        t21 = 2*t13
        t22 = 2*t16
        return np.pi**2.5*(-2*CDy*t19*t3 - 2*F3*p*t16**3*t4**(-3.5) - beta*t14*t20 - t10*t17 - t11*t13 + t11*t15*t22 + t11*t9 - t12*t7 + t16*t21*t8 + 3*t17*t18 + t20*t21 - t22*t8*t9 + t8 + t12*t14*t15/q**3)*np.exp(-alpha*rAB*t1 - delta*rCD*t3)
    if case_id == 139:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDx*gamma
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABy
        t9 = beta*t8
        t10 = CDx*CDy*t3
        t11 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t15 = 2*t14**2
        t16 = t1*t8
        t17 = CDy*delta
        t18 = t13*t14
        return np.pi**2.5*(F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 + t0*t7 + t11*t16*t17*t6 - t12*t15*t4 + t13 - t14*t16*t7 + 2*t17*t18 - t18*t9)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 140:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        return 2*np.pi**2.5*(ABy*CDx*F0*beta*delta*t4**(-0.5)/(p**2*q**2) - ABy*t1*t2*t6*t7 + CDx*t0*t3*t5*t6 - F2*t4**(-2.5)*t5*t7)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 141:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = CDx*delta
        t6 = CDz*gamma
        t7 = ABy*t6
        t8 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t9 = ABy*t8
        t10 = q**(-2.0)
        t11 = t4**(-1.5)
        t12 = F1*t1*t10*t11
        t13 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t14 = t13*t6
        t15 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t16 = t13*t8
        t17 = F2*t4**(-2.5)
        t18 = t15*t17*t2
        return 2*np.pi**2.5*(CDx*t16*t17*t3 + F0*beta*t4**(-0.5)*t5*t7/(p**2*q**3) + F1*t0*t10*t11*t14*t5 - F3*p*t15*t16*t4**(-3.5) - beta*t18*t9 - t12*t15*t7 + t12*t5*t9 - t14*t18)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 142:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDx*delta
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABy
        t9 = beta*t8
        t10 = CDx*CDy*t3
        t11 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t15 = 2*t14**2
        t16 = t1*t8
        t17 = CDy*gamma
        t18 = t13*t14
        return np.pi**2.5*(F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 - t0*t7 - t11*t16*t17*t6 + t12*t15*t4 + t13 + t14*t16*t7 - 2*t17*t18 - t18*t9)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 143:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = q**(-2.0)
        t7 = ABy*t6
        t8 = t1*t5*t7
        t9 = F0*beta*t4**(-0.5)/p**2
        t10 = 2*CDx**2*t3
        t11 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t12 = t11*t2
        t13 = F2*t4**(-2.5)
        t14 = t12*t13
        t15 = t0*t5
        t16 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t17 = 2*t16**2
        t18 = 2*CDx*t16
        t19 = t18*t8
        t20 = t14*t18
        return np.pi**2.5*(-ABy*beta*t13*t17*t2 + ABy*t10*t9/q**3 - F3*p*t11*t17*t4**(-3.5) + delta*t19 + delta*t20 - gamma*t19 - gamma*t20 + t10*t11*t15*t6 - t12*t15 + t14 - t7*t9 + t8)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 144:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = ABz*alpha
        t4 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        t5 = t2**(-1.5)
        t6 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        return 2*np.pi**2.5*t0*(-ABy*F0*beta*t1*t2**(-0.5)*t3/p**2 + ABy*F1*beta*t0*t5*t6 - F1*t0*t3*t4*t5 + F2*q*t2**(-2.5)*t4*t6)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 145:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = F1*t4*t6
        t8 = t2*t7
        t9 = ABy*ABz*t1
        t10 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t11 = t5**(-2.5)
        t12 = F2*t0*t10*t11
        t13 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t14 = t13**2
        t15 = 2*t13
        t16 = ABz*alpha
        return np.pi**2.5*(2*ABy*CDz*F1*beta*gamma*t13*t2*t4*t6 + 2*ABy*F2*beta*t0*t11*t14 - ABy*beta*t8 - 2*CDz*F0*gamma*t5**(-0.5)*t9/(p**3*q**2) + 2*CDz*F2*gamma*t0*t10*t11*t13 - 2*CDz*t10*t16*t3*t7 + 2*F3*q*t10*t14*t5**(-3.5) - t12*t15*t16 - t12 - t15*t8*t9)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 146:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = ABy*ABz*t1
        t8 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t9 = t5**(-2.5)
        t10 = F2*t0*t9
        t11 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t12 = t11**2
        t13 = ABz*alpha
        t14 = F1*t4*t6
        t15 = 2*t11
        return np.pi**2.5*(2*ABy*CDy*F1*beta*gamma*t2*t4*t6*t8 + 2*ABy*F2*beta*t0*t11*t8*t9 + ABz*F1*alpha*t2*t4*t6 - 2*CDy*F0*gamma*t5**(-0.5)*t7/(p**3*q**2) + 2*CDy*F2*gamma*t0*t11*t8*t9 - CDy*t13*t14*t15*t3 + 2*F3*q*t12*t5**(-3.5)*t8 - 2*t10*t12*t13 - t10*t8 - t14*t15*t2*t7)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 147:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = gamma*t1
        t3 = p + q
        t4 = ABy*beta
        t5 = ABz*alpha
        t6 = CDx*t5
        t7 = p**(-2.0)
        t8 = t3**(-1.5)
        t9 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        t10 = t5*t9
        t11 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        t12 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        t13 = t3**(-2.5)
        return 2*np.pi**2.5*(ABy*CDx*F1*beta*gamma*t1*t12*t7*t8 + ABy*F2*beta*t0*t12*t13*t9 + CDx*F2*gamma*t0*t11*t12*t13 - F0*gamma*t3**(-0.5)*t4*t6/(p**3*q**2) - F1*t1*t10*t4*t7*t8 - F1*t11*t2*t6*t7*t8 - F2*t0*t10*t11*t13 + F3*q*t11*t12*t3**(-3.5)*t9)*np.exp(-alpha*beta*rAB*t0 - delta*rCD*t2)
    if case_id == 148:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = ABy*beta
        t5 = p**(-2.0)
        t6 = p + q
        t7 = t6**(-1.5)
        t8 = F1*t5*t7
        t9 = t2*t8
        t10 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t11 = t6**(-2.5)
        t12 = F2*t0*t10*t11
        t13 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t14 = t13**2
        t15 = 2*t13
        t16 = t12*t15
        return np.pi**2.5*(2*ABy*ABz*CDz*F0*alpha*beta*delta*t6**(-0.5)/(p**3*q**2) - ABy*ABz*t1*t15*t9 + 2*ABy*F2*beta*t0*t11*t14 + 2*ABz*CDz*F1*alpha*delta*t10*t2*t5*t7 - ABz*alpha*t16 - CDz*delta*t16 - CDz*t15*t3*t4*t8 + 2*F3*q*t10*t14*t6**(-3.5) - t12 - t4*t9)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 149:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = q**(-2.0)
        t7 = ABy*beta
        t8 = p + q
        t9 = F1*t8**(-1.5)/p**2
        t10 = t6*t7*t9
        t11 = ABz*alpha
        t12 = t10*t11
        t13 = CDz*t10
        t14 = F0*t11*t7*t8**(-0.5)/p**3
        t15 = 2*CDz**2
        t16 = gamma*t15
        t17 = delta*t16
        t18 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t19 = t18**3
        t20 = F3*t8**(-3.5)
        t21 = 2*t20
        t22 = t3*t7
        t23 = t18*t22
        t24 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t25 = ABz*t24
        t26 = F2*t8**(-2.5)
        t27 = t26*t3
        t28 = t1*t27
        t29 = t0*t26
        t30 = CDz*t24
        t31 = t29*t30
        t32 = alpha*t25
        t33 = t32*t9
        t34 = t18**2
        t35 = 2*t34
        t36 = CDz*t4
        t37 = t29*t35
        t38 = CDz*gamma
        t39 = delta*t18
        t40 = 2*t38
        t41 = t18*t24
        t42 = t29*t41
        t43 = t21*t34
        t44 = t30*t43
        t45 = ABz*t41
        return np.pi**2.5*(-ABy*ABz*t2*t27*t35 + 2*CDz*t12*t39 + 2*F4*p*q*t19*t24*t8**(-4.5) + delta*t13 - delta*t44 - gamma*t13 - gamma*t3*t31 + gamma*t44 + 2*t1*t26*t36*t45 - t10*t16*t39 - t12*t18*t40 + t12 - t14*t6 - t15*t42*t5 + t17*t33*t6 + t19*t21*t7 - 3*t20*t41 + t22*t37*t38 - 3*t23*t29 + t23*t9 + t25*t28 - t28*t40*t45 - t3*t33 + t31*t4 - t32*t43 - t36*t37*t7 + t42 + t14*t17/q**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 150:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = t6**(-2.5)
        t8 = ABy*beta
        t9 = CDy*gamma
        t10 = p**(-2.0)
        t11 = q**(-2.0)
        t12 = t6**(-1.5)
        t13 = F1*t10*t11*t12
        t14 = t13*t8*t9
        t15 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t16 = t15**2
        t17 = t6**(-3.5)
        t18 = F3*t17
        t19 = t16*t18
        t20 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t21 = t20**2
        t22 = F2*t7
        t23 = t22*t3
        t24 = t0*t15*t23
        t25 = ABz*t20
        t26 = alpha*t25
        t27 = 2*t14
        t28 = CDz*t20
        t29 = delta*t28
        t30 = 2*t19
        t31 = 2*t15
        t32 = t23*t25
        t33 = t0*t28
        return np.pi**2.5*(2*ABy*ABz*CDy*CDz*F0*alpha*beta*delta*gamma*t6**(-0.5)/(p**3*q**3) + 2*ABy*ABz*CDz*F1*alpha*beta*delta*t10*t11*t12*t15 + 2*ABy*CDy*F2*beta*gamma*t0*t21*t3*t7 + 2*ABy*F3*beta*t15*t17*t21 - ABy*t2*t31*t32 + 2*ABz*CDy*CDz*F1*alpha*delta*gamma*t10*t11*t12*t15 + 2*ABz*CDz*F2*alpha*delta*t0*t16*t3*t7 - ABz*CDz*alpha*delta*t13 + ABz*F2*alpha*t0*t20*t3*t7 + 2*CDy*F3*gamma*t15*t17*t21 - CDy*t22*t31*t33*t5 + CDz*F2*delta*t0*t20*t3*t7 + (1/2)*F2*t0*t3*t7 + 2*F4*p*q*t16*t21*t6**(-4.5) - t1*t31*t32*t9 - t14 - t18*t21 - t19 - t22*t31*t33*t4*t8 - t24*t8 - t24*t9 - t26*t27 - t26*t30 - t27*t29 - t29*t30)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 151:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p**(-2.0)
        t7 = q**(-2.0)
        t8 = p + q
        t9 = t8**(-1.5)
        t10 = ABy*CDx*F1*beta*gamma*t6*t7*t9
        t11 = t8**(-2.5)
        t12 = F2*t11
        t13 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t14 = ABy*t13*t3
        t15 = t1*t12*t14
        t16 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t17 = CDx*t0*t12*t16
        t18 = t17*t4
        t19 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t20 = t19**2
        t21 = 2*t19
        t22 = ABz*alpha
        t23 = t21*t22
        t24 = CDz*delta
        t25 = t21*t24
        t26 = t8**(-3.5)
        t27 = F3*t13*t16*t26
        t28 = t21*t27
        return np.pi**2.5*(2*ABy*ABz*CDx*CDz*F0*alpha*beta*delta*gamma*t8**(-0.5)/(p**3*q**3) + 2*ABy*ABz*CDz*F1*alpha*beta*delta*t13*t6*t7*t9 + 2*ABy*CDx*F2*beta*gamma*t0*t11*t20*t3 + 2*ABy*F3*beta*t13*t20*t26 + 2*ABz*CDx*CDz*F1*alpha*delta*gamma*t16*t6*t7*t9 + 2*ABz*CDz*F2*alpha*delta*t0*t11*t13*t16*t3 - ABz*t12*t14*t2*t21 + 2*CDx*F3*gamma*t16*t20*t26 - CDz*t17*t21*t5 + 2*F4*p*q*t13*t16*t20*t8**(-4.5) - t10*t23 - t10*t25 - t10 - t15*t25 - t15 - t18*t23 - t18 - t22*t28 - t24*t28 - t27)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 152:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = ABz*alpha
        t5 = p + q
        t6 = F1*t5**(-1.5)/p**2
        t7 = t2*t6
        t8 = CDy*delta
        t9 = ABy*ABz*t1
        t10 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t11 = F2*t0*t5**(-2.5)
        t12 = t10*t11
        t13 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t14 = 2*t13**2
        t15 = 2*t13
        t16 = ABy*beta
        t17 = CDy*t3*t6
        t18 = t12*t15
        return np.pi**2.5*(2*F0*t5**(-0.5)*t8*t9/(p**3*q**2) + F3*q*t10*t14*t5**(-3.5) - 2*t10*t16*t17 - t11*t14*t4 - t12 + t15*t17*t4 - t15*t7*t9 + t16*t18 - t18*t8 + t4*t7)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 153:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F2*t6**(-2.5)
        t8 = t3*t7
        t9 = t0*t8
        t10 = ABy*beta
        t11 = CDy*t10
        t12 = delta*t11
        t13 = F1*t6**(-1.5)/(p**2*q**2)
        t14 = t12*t13
        t15 = CDz*gamma
        t16 = ABz*t15
        t17 = alpha*t16
        t18 = t13*t17
        t19 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t20 = t19**2
        t21 = F3*t6**(-3.5)
        t22 = t20*t21
        t23 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t24 = t23**2
        t25 = t21*t24
        t26 = ABz*t23
        t27 = t1*t8
        t28 = CDy*t19
        t29 = t0*t4*t7
        t30 = t10*t19
        t31 = t30*t9
        t32 = t15*t23
        t33 = 2*t24
        t34 = alpha*t26
        t35 = 2*t18
        t36 = 2*t32
        t37 = delta*t28
        t38 = 2*t25
        t39 = 2*t22
        t40 = 2*t26
        t41 = t28*t7
        return np.pi**2.5*(-ABy*t19*t2*t40*t8 - 2*CDz*t0*t23*t41*t5 + 2*F0*t12*t17*t6**(-0.5)/(p**3*q**3) + F4*p*q*t20*t33*t6**(-4.5) + t1*t4*t40*t41 - t11*t29*t33 + 2*t14*t34 - t14*t36 + t14 - 2*t16*t20*t27 + t18 - t22 - t25 + t26*t27 + t28*t29 - t30*t35 + t30*t38 + t31*t36 - t31 + t32*t39 - t32*t9 - t34*t39 + t35*t37 - t37*t38 + (1/2)*t9)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 154:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = ABz*alpha
        t7 = q**(-2.0)
        t8 = p + q
        t9 = F1*t8**(-1.5)/p**2
        t10 = t7*t9
        t11 = t10*t6
        t12 = ABy*beta
        t13 = CDy*t11
        t14 = F0*t12*t6*t8**(-0.5)/p**3
        t15 = 2*CDy**2
        t16 = delta*t15
        t17 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t18 = t17**3
        t19 = F3*t8**(-3.5)
        t20 = 2*t19
        t21 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t22 = ABy*t21
        t23 = beta*t22
        t24 = t3*t9
        t25 = F2*t8**(-2.5)
        t26 = t0*t25
        t27 = CDy*t21
        t28 = t26*t27
        t29 = t25*t3
        t30 = t1*t29
        t31 = t17*t6
        t32 = gamma*t3
        t33 = t17**2
        t34 = 2*t33
        t35 = CDy*t26*t34*t6
        t36 = 2*CDy
        t37 = t12*t31*t36
        t38 = gamma*t10
        t39 = t16*t38
        t40 = t17*t21
        t41 = t26*t40
        t42 = t20*t33
        t43 = t27*t42
        t44 = ABy*t36*t40
        return np.pi**2.5*(-ABy*ABz*t2*t29*t34 + 2*F4*p*q*t18*t21*t8**(-4.5) + delta*t10*t37 - delta*t13 - delta*t43 + gamma*t13 + gamma*t30*t44 + gamma*t43 + gamma*t14*t16/q**3 - t1*t25*t4*t44 + t11*t12 - t14*t7 - t15*t41*t5 - t18*t20*t6 - 3*t19*t40 - t22*t30 + t23*t24 - t23*t39 + t23*t42 - t24*t31 + 3*t26*t3*t31 - t28*t32 + t28*t4 + t31*t39 - t32*t35 + t35*t4 - t37*t38 + t41)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 155:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABz*CDx
        t9 = alpha*gamma*t8
        t10 = t7*t9
        t11 = CDy*delta
        t12 = ABy*beta
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t1*t14
        t16 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t17 = ABz*t16
        t18 = t17*t3
        t19 = t15*t18
        t20 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t21 = CDx*t20
        t22 = t0*t14
        t23 = t21*t22
        t24 = t23*t4
        t25 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t26 = 2*t25**2
        t27 = 2*t25
        t28 = t12*t27
        t29 = alpha*t17
        t30 = t13*t7
        t31 = gamma*t21
        t32 = t11*t27
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t27*t35
        return np.pi**2.5*(-ABy*t14*t18*t2*t27 - CDy*t23*t27*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) - t10*t28 + t10*t32 + t10 - t11*t37 + t12*t37 - t13*t22*t3*t33 - t15*t26*t4*t8 + t19*t32 + t19 + t24*t28 - t24 + t29*t30 - t29*t36 - t30*t31 + t31*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 156:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = ABz*alpha
        t6 = CDx*t5
        t7 = ABy*beta
        t8 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t9 = F1*t4**(-1.5)/p**2
        t10 = t3*t9
        t11 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t12 = t11*t5
        t13 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t14 = CDx*t13
        t15 = t11*t13
        t16 = F2*t4**(-2.5)
        t17 = t0*t16*t8
        return 2*np.pi**2.5*(ABy*t1*t15*t16 + F0*delta*t4**(-0.5)*t6*t7/(p**3*q**2) + F3*q*t15*t4**(-3.5)*t8 - delta*t14*t17 - t10*t14*t7 + t10*t6*t8 - t12*t17 - t12*t2*t7*t9)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 157:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABy*CDx
        t9 = beta*delta*t8
        t10 = t7*t9
        t11 = ABz*alpha
        t12 = CDz*gamma
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t14*t4
        t16 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t17 = CDx*t16
        t18 = t0*t17
        t19 = t15*t18
        t20 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t21 = ABy*t20
        t22 = t14*t3
        t23 = t21*t22
        t24 = t1*t23
        t25 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t26 = 2*t25**2
        t27 = 2*t25
        t28 = t11*t27
        t29 = beta*t21
        t30 = t13*t7
        t31 = t12*t27
        t32 = delta*t17
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t27*t35
        return np.pi**2.5*(-ABz*t2*t23*t27 - CDz*t14*t18*t27*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) - t0*t13*t22*t33 - t1*t15*t26*t8 + t10*t28 - t10*t31 + t10 - t11*t37 + t12*t37 + t19*t28 + t19 + t24*t31 - t24 - t29*t30 + t29*t36 + t30*t32 - t32*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 158:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABz*CDx
        t9 = alpha*delta*t8
        t10 = t7*t9
        t11 = ABy*beta
        t12 = CDy*gamma
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t1*t14
        t16 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t17 = ABz*t16
        t18 = t17*t3
        t19 = t15*t18
        t20 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t21 = CDx*t20
        t22 = t0*t14
        t23 = t21*t22
        t24 = t23*t4
        t25 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t26 = 2*t25**2
        t27 = 2*t25
        t28 = t11*t27
        t29 = alpha*t17
        t30 = t13*t7
        t31 = delta*t21
        t32 = t12*t27
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t27*t35
        return np.pi**2.5*(-ABy*t14*t18*t2*t27 - CDy*t23*t27*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) + t10*t28 + t10*t32 - t10 + t11*t37 + t12*t37 + t13*t22*t3*t33 + t15*t26*t4*t8 - t19*t32 + t19 - t24*t28 + t24 - t29*t30 - t29*t36 - t30*t31 - t31*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 159:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = q**(-2.0)
        t7 = p + q
        t8 = F1*t7**(-1.5)/p**2
        t9 = t6*t8
        t10 = ABy*beta
        t11 = ABz*alpha*t10
        t12 = t11*t9
        t13 = F0*t11*t7**(-0.5)/p**3
        t14 = CDx**2
        t15 = 2*delta
        t16 = gamma*t14*t15
        t17 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t18 = t10*t17
        t19 = t18*t3
        t20 = F2*t7**(-2.5)
        t21 = t20*t3
        t22 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t23 = ABz*t22
        t24 = t1*t23
        t25 = t21*t24
        t26 = t0*t20
        t27 = t19*t26
        t28 = alpha*t23
        t29 = t28*t8
        t30 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t31 = 2*t30**2
        t32 = CDx*t30
        t33 = 2*gamma
        t34 = t32*t33
        t35 = t17*t22
        t36 = F3*t7**(-3.5)
        t37 = t35*t36
        t38 = t26*t35
        t39 = t31*t36
        t40 = 2*t32*t4
        t41 = t32*t37
        return np.pi**2.5*(-ABy*ABz*t2*t21*t31 + F4*p*q*t31*t35*t7**(-4.5) + t12*t15*t32 - t12*t34 + t12 - t13*t6 - 2*t14*t38*t5 - t15*t41 - t16*t18*t9 + t16*t29*t6 - t18*t26*t40 + t18*t39 + t19*t8 + t20*t24*t40 - t25*t34 + t25 + t27*t34 - t27 - t28*t39 - t29*t3 + t33*t41 - t37 + t38 + t13*t16/q**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 160:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = p + q
        t4 = t3**(-1.5)
        t5 = F1*t0*t4
        t6 = t3**(-0.5)
        t7 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        return np.pi**2.5*t0*(-2*ABy**2*F0*t1*t2*t6/p**2 + 2*ABy*F1*beta*t0*t4*t7 - 2*ABy*alpha*t5*t7 + F0*t0*t2*t6 + 2*F2*q*t3**(-2.5)*t7**2 - t5)*np.exp(-delta*gamma*rCD*t2 - rAB*t0*t1)
    if case_id == 161:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = p**(-2.0)
        t8 = CDz*t3*t6*t7
        t9 = q**(-2.0)
        t10 = t4**(-0.5)
        t11 = 2*ABy**2*t1
        t12 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t13 = t4**(-2.5)
        t14 = F2*t0*t12*t13
        t15 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t16 = t15**2
        t17 = 2*ABy*t15
        return np.pi**2.5*(2*ABy*CDz*F1*beta*gamma*t15*t2*t5*t7 + 2*ABy*F2*beta*t0*t12*t13*t15 + CDz*F0*gamma*t10*t7*t9 - CDz*F0*gamma*t10*t11*t9/p**3 + 2*CDz*F2*gamma*t0*t13*t16 + F1*t0*t12*t2*t5 + 2*F3*q*t12*t16*t4**(-3.5) - alpha*t14*t17 - alpha*t17*t8 - t11*t12*t2*t6*t7 - t14 - t8)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 162:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = t2*t6
        t8 = p**(-2.0)
        t9 = ABy*t8
        t10 = t3*t6
        t11 = q**(-2.0)
        t12 = t4**(-0.5)
        t13 = ABy**2*beta
        t14 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t15 = t4**(-2.5)
        t16 = F2*t15
        t17 = t14**2
        t18 = 2*alpha*t14
        return np.pi**2.5*(2*ABy*CDy*F1*beta*gamma*t14*t2*t5*t8 + ABy*F1*alpha*t2*t5*t8 + 2*ABy*F2*beta*t0*t15*t17 - 2*ABy*t1*t16*t17 - 2*CDy*F0*alpha*gamma*t11*t12*t13/p**3 + CDy*F0*gamma*t11*t12*t8 + 2*CDy*F2*gamma*t0*t15*t17 - CDy*t10*t18*t9 - CDy*t10*t8 + F1*t0*t14*t2*t5 + 2*F3*q*t14**3*t4**(-3.5) - beta*t7*t9 - 3*t0*t14*t16 - t13*t18*t7*t8)*np.exp(-beta*rAB*t1 - delta*rCD*t3)
    if case_id == 163:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = p**(-2.0)
        t8 = CDx*t3*t6*t7
        t9 = q**(-2.0)
        t10 = t4**(-0.5)
        t11 = 2*ABy**2*t1
        t12 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t13 = t4**(-2.5)
        t14 = F2*t0*t12*t13
        t15 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t16 = t15**2
        t17 = 2*ABy*t15
        return np.pi**2.5*(2*ABy*CDx*F1*beta*gamma*t15*t2*t5*t7 + 2*ABy*F2*beta*t0*t12*t13*t15 + CDx*F0*gamma*t10*t7*t9 - CDx*F0*gamma*t10*t11*t9/p**3 + 2*CDx*F2*gamma*t0*t13*t16 + F1*t0*t12*t2*t5 + 2*F3*q*t12*t16*t4**(-3.5) - alpha*t14*t17 - alpha*t17*t8 - t11*t12*t2*t6*t7 - t14 - t8)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 164:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = p**(-2.0)
        t7 = CDz*t6
        t8 = t3*t5*t7
        t9 = F0*delta*t4**(-0.5)/q**2
        t10 = 2*ABy**2*t1
        t11 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t12 = t0*t11
        t13 = F2*t4**(-2.5)
        t14 = t12*t13
        t15 = t2*t5
        t16 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t17 = 2*t16**2
        t18 = 2*ABy*t16
        t19 = t18*t8
        t20 = t14*t18
        return np.pi**2.5*(-CDz*delta*t0*t13*t17 + CDz*t10*t9/p**3 + F3*q*t11*t17*t4**(-3.5) + alpha*t19 - alpha*t20 - beta*t19 + beta*t20 - t10*t11*t15*t6 + t12*t15 - t14 - t7*t9 + t8)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 165:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = gamma*t3
        t5 = q**(-2.0)
        t6 = (1/2)*t5
        t7 = p**(-2.0)
        t8 = p + q
        t9 = F0*t8**(-0.5)
        t10 = t7*t9
        t11 = (1/2)*t2
        t12 = F1*t8**(-1.5)
        t13 = t12*t7
        t14 = t0*t12
        t15 = F2*t8**(-2.5)
        t16 = t0*t15
        t17 = ABy**2*t1
        t18 = t17*t5
        t19 = t13*t18
        t20 = CDz**2
        t21 = delta*t20
        t22 = gamma*t5
        t23 = t13*t22
        t24 = t9/p**3
        t25 = gamma/q**3
        t26 = 2*delta
        t27 = t20*t26
        t28 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t29 = t28**2
        t30 = F3*t8**(-3.5)
        t31 = t29*t30
        t32 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t33 = t32**2
        t34 = t30*t33
        t35 = t16*t29
        t36 = t2*t33
        t37 = ABy*t28
        t38 = beta*t37
        t39 = t13*t2
        t40 = alpha*t37
        t41 = t16*t2
        t42 = t40*t41
        t43 = CDz*t32
        t44 = t14*t43
        t45 = t16*t3*t43
        t46 = t38*t41
        t47 = t23*t27
        t48 = 2*gamma
        t49 = t43*t48
        t50 = 2*t34
        t51 = t31*t43
        t52 = 2*t45
        return np.pi**2.5*(2*F4*p*q*t29*t33*t8**(-4.5) - delta*t44*t5 - gamma*t41*t43 - t10*t21*t25 + t10*t6 - t11*t13 + t11*t16 - t14*t6 + t15*t36 - 2*t16*t17*t36 + t17*t24*t25*t27 - t18*t24 + t19*t26*t43 - t19*t49 + t19 - 2*t20*t35*t4 + t21*t23 + t22*t44 - t26*t51 - t31 - t34 + t35 + t38*t39 - t38*t47 + t38*t50 - t38*t52 - t39*t40 + t40*t47 - t40*t50 + t40*t52 - t42*t49 + t42 + t45 + t46*t49 - t46 + t48*t51)*np.exp(-rAB*t0*t1 - rCD*t4)
    if case_id == 166:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/q**2
        t8 = p**(-2.0)
        t9 = CDz*delta
        t10 = t8*t9
        t11 = t10*t7
        t12 = ABy*t11
        t13 = CDy*gamma
        t14 = F0*t13*t6**(-0.5)/q**3
        t15 = 2*ABy**2
        t16 = alpha*t15
        t17 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t18 = t17**3
        t19 = F3*t6**(-3.5)
        t20 = 2*t19
        t21 = F2*t6**(-2.5)
        t22 = t21*t3
        t23 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t24 = ABy*t23
        t25 = t22*t24
        t26 = CDy*t23
        t27 = gamma*t26
        t28 = t0*t7
        t29 = beta*t0
        t30 = t0*t21
        t31 = t17*t9
        t32 = t17**2
        t33 = 2*t32
        t34 = t22*t33*t9
        t35 = ABy*t1
        t36 = ABy*t29
        t37 = 2*ABy*t13*t31
        t38 = beta*t7*t8
        t39 = t16*t38
        t40 = t17*t23
        t41 = t22*t40
        t42 = t20*t32
        t43 = t24*t42
        t44 = 2*CDy*t21*t4*t40
        return np.pi**2.5*(-CDy*CDz*t30*t33*t5 + 2*F4*p*q*t18*t23*t6**(-4.5) - alpha*t12 + alpha*t37*t7*t8 - alpha*t43 + beta*t12 + beta*t43 + beta*t14*t16*t9/p**3 + 3*t0*t22*t31 + t1*t25 - t10*t14 + t11*t13 - t15*t2*t41 - t18*t20*t9 - 3*t19*t40 - t25*t29 - t26*t30*t4 + t27*t28 - t27*t39 + t27*t42 - t28*t31 + t31*t39 + t34*t35 - t34*t36 - t35*t44 + t36*t44 - t37*t38 + t41)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 167:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p**(-2.0)
        t7 = p + q
        t8 = F1*t7**(-1.5)/q**2
        t9 = t6*t8
        t10 = CDx*gamma
        t11 = CDz*delta*t10
        t12 = t11*t9
        t13 = F0*t11*t7**(-0.5)/q**3
        t14 = ABy**2
        t15 = 2*alpha
        t16 = beta*t14*t15
        t17 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t18 = t10*t17
        t19 = t0*t18
        t20 = F2*t7**(-2.5)
        t21 = t0*t20
        t22 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t23 = CDz*t22
        t24 = t23*t4
        t25 = t21*t24
        t26 = t20*t3
        t27 = t19*t26
        t28 = delta*t23
        t29 = t28*t8
        t30 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t31 = 2*t30**2
        t32 = ABy*t30
        t33 = 2*beta
        t34 = t32*t33
        t35 = t17*t22
        t36 = F3*t7**(-3.5)
        t37 = t35*t36
        t38 = t26*t35
        t39 = t31*t36
        t40 = 2*t1*t32
        t41 = t32*t37
        return np.pi**2.5*(-CDx*CDz*t21*t31*t5 + F4*p*q*t31*t35*t7**(-4.5) - t0*t29 + t12*t15*t32 - t12*t34 + t12 - t13*t6 - 2*t14*t2*t38 - t15*t41 - t16*t18*t9 + t16*t29*t6 - t18*t26*t40 + t18*t39 + t19*t8 + t20*t24*t40 - t25*t34 + t25 + t27*t34 - t27 - t28*t39 + t33*t41 - t37 + t38 + t13*t16/p**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 168:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = t2*t5
        t7 = p**(-2.0)
        t8 = ABy*t7
        t9 = t6*t8
        t10 = CDy*t7
        t11 = t3*t5
        t12 = F0*delta*t4**(-0.5)/q**2
        t13 = ABy**2*alpha
        t14 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t15 = t0*t14
        t16 = F2*t4**(-2.5)
        t17 = 2*t14**2*t16
        t18 = ABy*t17
        t19 = CDy*t11*t8
        t20 = 2*beta*t14
        return np.pi**2.5*(2*CDy*beta*t12*t13/p**3 - CDy*delta*t0*t17 + 2*F3*q*t14**3*t4**(-3.5) + 2*alpha*t14*t19 + alpha*t9 + beta*t0*t18 - beta*t9 - t1*t18 + t10*t11 - t10*t12 - t13*t20*t6*t7 - 3*t15*t16 + t15*t6 - t19*t20)*np.exp(-beta*rAB*t1 - gamma*rCD*t3)
    if case_id == 169:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/q**2
        t8 = p**(-2.0)
        t9 = CDz*gamma
        t10 = t8*t9
        t11 = t10*t7
        t12 = ABy*t11
        t13 = alpha*t12
        t14 = CDy*delta
        t15 = beta*t12
        t16 = F0*t14*t6**(-0.5)/q**3
        t17 = 2*ABy**2
        t18 = alpha*beta*t17
        t19 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t20 = t19**3
        t21 = F3*t6**(-3.5)
        t22 = 2*t21
        t23 = F2*t6**(-2.5)
        t24 = t23*t3
        t25 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t26 = ABy*t25
        t27 = t24*t26
        t28 = CDy*t25
        t29 = t0*t23
        t30 = t29*t4
        t31 = t0*t9
        t32 = t19*t31
        t33 = delta*t28
        t34 = t33*t7
        t35 = ABy*t1
        t36 = t19**2
        t37 = 2*t36
        t38 = t24*t37
        t39 = ABy*beta
        t40 = 2*t14*t19
        t41 = t19*t25
        t42 = t24*t41
        t43 = t22*t36
        t44 = t26*t43
        t45 = 2*CDy*t41
        return np.pi**2.5*(-CDy*CDz*t29*t37*t5 + 2*F4*p*q*t20*t25*t6**(-4.5) - alpha*t44 - beta*t0*t27 + beta*t44 - t0*t34 + t1*t27 - t10*t16 + t11*t14 - t11*t18*t19 + t13*t40 + t13 - t15*t40 - t15 - t17*t2*t42 + t18*t34*t8 + t20*t22*t9 - 3*t21*t41 + t23*t35*t4*t45 - 3*t24*t32 + t28*t30 - t30*t39*t45 + t31*t38*t39 + t32*t7 - t33*t43 - t35*t38*t9 + t42 + t16*t18*t9/p**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 170:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = t2*t3
        t5 = q**(-2.0)
        t6 = (1/2)*t5
        t7 = p**(-2.0)
        t8 = p + q
        t9 = F0*t8**(-0.5)
        t10 = t7*t9
        t11 = F1*t8**(-1.5)
        t12 = t11*t7
        t13 = t12*t2
        t14 = t0*t11
        t15 = t8**(-2.5)
        t16 = F2*t0*t15*t2
        t17 = ABy*alpha
        t18 = t12*t5
        t19 = CDy*t18
        t20 = t17*t19
        t21 = ABy*beta
        t22 = t19*t21
        t23 = ABy**2*t1
        t24 = t18*t23
        t25 = CDy**2
        t26 = t25*t3
        t27 = t18*t26
        t28 = t23*t9/p**3
        t29 = t26/q**3
        t30 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t31 = t30**2
        t32 = F3*t8**(-3.5)
        t33 = F2*t15*t31
        t34 = t0*t33
        t35 = 2*t30**3*t32
        t36 = CDy*t35
        t37 = t13*t30
        t38 = CDy*t30
        t39 = gamma*t38
        t40 = t14*t5
        t41 = 3*t16
        t42 = t17*t30
        t43 = t21*t30
        t44 = delta*t38
        t45 = 2*t34
        t46 = t2*t45
        t47 = CDy*t46
        t48 = t17*t47
        t49 = t21*t47
        t50 = 2*t27
        t51 = 2*t24
        return np.pi**2.5*(2*F4*p*q*t30**4*t8**(-4.5) - delta*t20 + delta*t22 - delta*t36 + delta*t48 - delta*t49 + gamma*t20 - gamma*t22 + gamma*t36 - gamma*t48 + gamma*t49 - t10*t29 + t10*t6 - 1/2*t13 - t14*t6 + (3/2)*t16 - t17*t35 - t17*t37 + t2*t33 + t21*t35 + t21*t37 - t23*t46 + t24 - t25*t4*t45 + t27 + 2*t28*t29 - t28*t5 - 6*t31*t32 + t34 + t39*t40 - t39*t41 - t39*t51 - t40*t44 + t41*t42 - t41*t43 + t41*t44 + t42*t50 - t43*t50 + t44*t51)*np.exp(-rAB*t0*t1 - rCD*t4)
    if case_id == 171:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/q**2
        t8 = p**(-2.0)
        t9 = CDx*gamma
        t10 = t8*t9
        t11 = t10*t7
        t12 = ABy*t11
        t13 = alpha*t12
        t14 = CDy*delta
        t15 = beta*t12
        t16 = F0*t14*t6**(-0.5)/q**3
        t17 = 2*ABy**2
        t18 = alpha*beta*t17
        t19 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t20 = t19**3
        t21 = F3*t6**(-3.5)
        t22 = 2*t21
        t23 = F2*t6**(-2.5)
        t24 = t23*t3
        t25 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t26 = ABy*t25
        t27 = t24*t26
        t28 = t0*t9
        t29 = t19*t28
        t30 = CDy*t25
        t31 = t0*t23
        t32 = t31*t4
        t33 = delta*t30
        t34 = t33*t7
        t35 = ABy*t1
        t36 = t19**2
        t37 = 2*t36
        t38 = t24*t37
        t39 = ABy*beta
        t40 = 2*t14*t19
        t41 = t19*t25
        t42 = t24*t41
        t43 = t22*t36
        t44 = t26*t43
        t45 = 2*CDy*t41
        return np.pi**2.5*(-CDx*CDy*t31*t37*t5 + 2*F4*p*q*t20*t25*t6**(-4.5) - alpha*t44 - beta*t0*t27 + beta*t44 - t0*t34 + t1*t27 - t10*t16 + t11*t14 - t11*t18*t19 + t13*t40 + t13 - t15*t40 - t15 - t17*t2*t42 + t18*t34*t8 + t20*t22*t9 - 3*t21*t41 + t23*t35*t4*t45 - 3*t24*t29 + t28*t38*t39 + t29*t7 + t30*t32 - t32*t39*t45 - t33*t43 - t35*t38*t9 + t42 + t16*t18*t9/p**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 172:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = p**(-2.0)
        t7 = CDx*t6
        t8 = t3*t5*t7
        t9 = F0*delta*t4**(-0.5)/q**2
        t10 = 2*ABy**2*t1
        t11 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t12 = t0*t11
        t13 = F2*t4**(-2.5)
        t14 = t12*t13
        t15 = t2*t5
        t16 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t17 = 2*t16**2
        t18 = 2*ABy*t16
        t19 = t18*t8
        t20 = t14*t18
        return np.pi**2.5*(-CDx*delta*t0*t13*t17 + CDx*t10*t9/p**3 + F3*q*t11*t17*t4**(-3.5) + alpha*t19 - alpha*t20 - beta*t19 + beta*t20 - t10*t11*t15*t6 + t12*t15 - t14 - t7*t9 + t8)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 173:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p**(-2.0)
        t7 = p + q
        t8 = F1*t7**(-1.5)/q**2
        t9 = t6*t8
        t10 = CDz*gamma
        t11 = CDx*delta*t10
        t12 = t11*t9
        t13 = F0*t11*t7**(-0.5)/q**3
        t14 = ABy**2
        t15 = 2*alpha
        t16 = beta*t14*t15
        t17 = F2*t7**(-2.5)
        t18 = t0*t17
        t19 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t20 = CDx*t19
        t21 = t20*t4
        t22 = t18*t21
        t23 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t24 = t10*t23
        t25 = t0*t24
        t26 = delta*t20
        t27 = t26*t8
        t28 = t17*t3
        t29 = t25*t28
        t30 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t31 = 2*t30**2
        t32 = ABy*t30
        t33 = t15*t32
        t34 = 2*beta
        t35 = t19*t23
        t36 = F3*t7**(-3.5)
        t37 = t35*t36
        t38 = t28*t35
        t39 = t31*t36
        t40 = 2*t1*t32
        t41 = t32*t37
        return np.pi**2.5*(-CDx*CDz*t18*t31*t5 + F4*p*q*t31*t35*t7**(-4.5) - t0*t27 - t12*t32*t34 + t12*t33 + t12 - t13*t6 - 2*t14*t2*t38 - t15*t41 - t16*t24*t9 + t16*t27*t6 - t17*t21*t40 + t22*t33 + t22 + t24*t28*t40 + t24*t39 + t25*t8 - t26*t39 - t29*t33 - t29 + t34*t41 - t37 + t38 + t13*t16/p**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 174:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/q**2
        t8 = p**(-2.0)
        t9 = CDx*delta
        t10 = t8*t9
        t11 = t10*t7
        t12 = ABy*t11
        t13 = CDy*gamma
        t14 = F0*t13*t6**(-0.5)/q**3
        t15 = 2*ABy**2
        t16 = alpha*t15
        t17 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t18 = t17**3
        t19 = F3*t6**(-3.5)
        t20 = 2*t19
        t21 = F2*t6**(-2.5)
        t22 = t21*t3
        t23 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t24 = ABy*t23
        t25 = t22*t24
        t26 = CDy*t23
        t27 = gamma*t26
        t28 = t0*t7
        t29 = beta*t0
        t30 = t17*t9
        t31 = t0*t21
        t32 = t17**2
        t33 = 2*t32
        t34 = t22*t33*t9
        t35 = ABy*t1
        t36 = ABy*t29
        t37 = 2*ABy*t13*t30
        t38 = beta*t7*t8
        t39 = t16*t38
        t40 = t17*t23
        t41 = t22*t40
        t42 = t20*t32
        t43 = t24*t42
        t44 = 2*CDy*t21*t4*t40
        return np.pi**2.5*(-CDx*CDy*t31*t33*t5 + 2*F4*p*q*t18*t23*t6**(-4.5) - alpha*t12 + alpha*t37*t7*t8 - alpha*t43 + beta*t12 + beta*t43 + beta*t14*t16*t9/p**3 + 3*t0*t22*t30 + t1*t25 - t10*t14 + t11*t13 - t15*t2*t41 - t18*t20*t9 - 3*t19*t40 - t25*t29 - t26*t31*t4 + t27*t28 - t27*t39 + t27*t42 - t28*t30 + t30*t39 + t34*t35 - t34*t36 - t35*t44 + t36*t44 - t37*t38 + t41)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 175:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = gamma*t3
        t5 = q**(-2.0)
        t6 = (1/2)*t5
        t7 = p**(-2.0)
        t8 = p + q
        t9 = F0*t8**(-0.5)
        t10 = t7*t9
        t11 = (1/2)*t2
        t12 = F1*t8**(-1.5)
        t13 = t12*t7
        t14 = t0*t12
        t15 = F2*t8**(-2.5)
        t16 = t0*t15
        t17 = ABy**2*t1
        t18 = t17*t5
        t19 = t13*t18
        t20 = CDx**2
        t21 = delta*t20
        t22 = gamma*t5
        t23 = t13*t22
        t24 = t9/p**3
        t25 = gamma/q**3
        t26 = 2*delta
        t27 = t20*t26
        t28 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t29 = t28**2
        t30 = F3*t8**(-3.5)
        t31 = t29*t30
        t32 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t33 = t32**2
        t34 = t30*t33
        t35 = t16*t33
        t36 = t2*t29
        t37 = ABy*t32
        t38 = beta*t37
        t39 = t13*t2
        t40 = alpha*t37
        t41 = t16*t2
        t42 = t40*t41
        t43 = CDx*t28
        t44 = t14*t43
        t45 = t16*t3*t43
        t46 = t38*t41
        t47 = t23*t27
        t48 = 2*gamma
        t49 = t43*t48
        t50 = 2*t31
        t51 = t34*t43
        t52 = 2*t45
        return np.pi**2.5*(2*F4*p*q*t29*t33*t8**(-4.5) - delta*t44*t5 - gamma*t41*t43 - t10*t21*t25 + t10*t6 - t11*t13 + t11*t16 - t14*t6 + t15*t36 - 2*t16*t17*t36 + t17*t24*t25*t27 - t18*t24 + t19*t26*t43 - t19*t49 + t19 - 2*t20*t35*t4 + t21*t23 + t22*t44 - t26*t51 - t31 - t34 + t35 + t38*t39 - t38*t47 + t38*t50 - t38*t52 - t39*t40 + t40*t47 - t40*t50 + t40*t52 - t42*t49 + t42 + t45 + t46*t49 - t46 + t48*t51)*np.exp(-rAB*t0*t1 - rCD*t4)
    if case_id == 176:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = ABx*alpha
        t4 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        t5 = t2**(-1.5)
        t6 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        return 2*np.pi**2.5*t0*(-ABy*F0*beta*t1*t2**(-0.5)*t3/p**2 + ABy*F1*beta*t0*t5*t6 - F1*t0*t3*t4*t5 + F2*q*t2**(-2.5)*t4*t6)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 177:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = gamma*t1
        t3 = p + q
        t4 = ABy*beta
        t5 = ABx*alpha
        t6 = CDz*t5
        t7 = p**(-2.0)
        t8 = t3**(-1.5)
        t9 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        t10 = t5*t9
        t11 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        t12 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        t13 = t3**(-2.5)
        return 2*np.pi**2.5*(ABy*CDz*F1*beta*gamma*t1*t12*t7*t8 + ABy*F2*beta*t0*t12*t13*t9 + CDz*F2*gamma*t0*t11*t12*t13 - F0*gamma*t3**(-0.5)*t4*t6/(p**3*q**2) - F1*t1*t10*t4*t7*t8 - F1*t11*t2*t6*t7*t8 - F2*t0*t10*t11*t13 + F3*q*t11*t12*t3**(-3.5)*t9)*np.exp(-alpha*beta*rAB*t0 - delta*rCD*t2)
    if case_id == 178:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = ABx*ABy*t1
        t8 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t9 = t5**(-2.5)
        t10 = F2*t0*t9
        t11 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t12 = t11**2
        t13 = ABx*alpha
        t14 = F1*t4*t6
        t15 = 2*t11
        return np.pi**2.5*(ABx*F1*alpha*t2*t4*t6 + 2*ABy*CDy*F1*beta*gamma*t2*t4*t6*t8 + 2*ABy*F2*beta*t0*t11*t8*t9 - 2*CDy*F0*gamma*t5**(-0.5)*t7/(p**3*q**2) + 2*CDy*F2*gamma*t0*t11*t8*t9 - CDy*t13*t14*t15*t3 + 2*F3*q*t12*t5**(-3.5)*t8 - 2*t10*t12*t13 - t10*t8 - t14*t15*t2*t7)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 179:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = F1*t4*t6
        t8 = t2*t7
        t9 = ABx*ABy*t1
        t10 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t11 = t5**(-2.5)
        t12 = F2*t0*t10*t11
        t13 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t14 = t13**2
        t15 = 2*t13
        t16 = ABx*alpha
        return np.pi**2.5*(2*ABy*CDx*F1*beta*gamma*t13*t2*t4*t6 + 2*ABy*F2*beta*t0*t11*t14 - ABy*beta*t8 - 2*CDx*F0*gamma*t5**(-0.5)*t9/(p**3*q**2) + 2*CDx*F2*gamma*t0*t10*t11*t13 - 2*CDx*t10*t16*t3*t7 + 2*F3*q*t10*t14*t5**(-3.5) - t12*t15*t16 - t12 - t15*t8*t9)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 180:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = ABx*alpha
        t6 = CDz*t5
        t7 = ABy*beta
        t8 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t9 = F1*t4**(-1.5)/p**2
        t10 = t3*t9
        t11 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t12 = t11*t5
        t13 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t14 = CDz*t13
        t15 = t11*t13
        t16 = F2*t4**(-2.5)
        t17 = t0*t16*t8
        return 2*np.pi**2.5*(ABy*t1*t15*t16 + F0*delta*t4**(-0.5)*t6*t7/(p**3*q**2) + F3*q*t15*t4**(-3.5)*t8 - delta*t14*t17 - t10*t14*t7 + t10*t6*t8 - t12*t17 - t12*t2*t7*t9)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 181:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = q**(-2.0)
        t7 = p + q
        t8 = F1*t7**(-1.5)/p**2
        t9 = t6*t8
        t10 = ABy*beta
        t11 = ABx*alpha*t10
        t12 = t11*t9
        t13 = F0*t11*t7**(-0.5)/p**3
        t14 = CDz**2
        t15 = 2*delta
        t16 = gamma*t14*t15
        t17 = F2*t7**(-2.5)
        t18 = t17*t3
        t19 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t20 = ABx*t19
        t21 = t1*t20
        t22 = t18*t21
        t23 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t24 = t10*t23
        t25 = t24*t3
        t26 = alpha*t20
        t27 = t26*t8
        t28 = t0*t17
        t29 = t25*t28
        t30 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t31 = 2*t30**2
        t32 = CDz*t30
        t33 = t15*t32
        t34 = 2*gamma
        t35 = t19*t23
        t36 = F3*t7**(-3.5)
        t37 = t35*t36
        t38 = t28*t35
        t39 = t31*t36
        t40 = 2*t32*t4
        t41 = t32*t37
        return np.pi**2.5*(-ABx*ABy*t18*t2*t31 + F4*p*q*t31*t35*t7**(-4.5) - t12*t32*t34 + t12*t33 + t12 - t13*t6 - 2*t14*t38*t5 - t15*t41 - t16*t24*t9 + t16*t27*t6 - t17*t21*t40 + t22*t33 + t22 + t24*t28*t40 + t24*t39 + t25*t8 - t26*t39 - t27*t3 - t29*t33 - t29 + t34*t41 - t37 + t38 + t13*t16/q**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 182:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABx*CDz
        t9 = alpha*delta*t8
        t10 = t7*t9
        t11 = ABy*beta
        t12 = CDy*gamma
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t1*t14
        t16 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t17 = ABx*t16
        t18 = t17*t3
        t19 = t15*t18
        t20 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t21 = CDz*t20
        t22 = t0*t14
        t23 = t21*t22
        t24 = t23*t4
        t25 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t26 = 2*t25**2
        t27 = alpha*t17
        t28 = t13*t7
        t29 = 2*t25
        t30 = t11*t29
        t31 = t12*t29
        t32 = delta*t21
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t29*t35
        return np.pi**2.5*(-ABy*t14*t18*t2*t29 - CDy*t23*t29*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) + t10*t30 + t10*t31 - t10 + t11*t37 + t12*t37 + t13*t22*t3*t33 + t15*t26*t4*t8 - t19*t31 + t19 - t24*t30 + t24 - t27*t28 - t27*t36 - t28*t32 - t32*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 183:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABy*CDz
        t9 = beta*delta*t8
        t10 = t7*t9
        t11 = ABx*alpha
        t12 = CDx*gamma
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t14*t4
        t16 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t17 = CDz*t16
        t18 = t0*t17
        t19 = t15*t18
        t20 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t21 = ABy*t20
        t22 = t14*t3
        t23 = t21*t22
        t24 = t1*t23
        t25 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t26 = 2*t25**2
        t27 = beta*t21
        t28 = t13*t7
        t29 = 2*t25
        t30 = t11*t29
        t31 = delta*t17
        t32 = t12*t29
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t29*t35
        return np.pi**2.5*(-ABx*t2*t23*t29 - CDx*t14*t18*t29*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) - t0*t13*t22*t33 - t1*t15*t26*t8 + t10*t30 - t10*t32 + t10 - t11*t37 + t12*t37 + t19*t30 + t19 + t24*t32 - t24 - t27*t28 + t27*t36 + t28*t31 - t31*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 184:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = ABx*alpha
        t5 = p + q
        t6 = F1*t5**(-1.5)/p**2
        t7 = t2*t6
        t8 = CDy*delta
        t9 = ABx*ABy*t1
        t10 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t11 = F2*t0*t5**(-2.5)
        t12 = t10*t11
        t13 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t14 = 2*t13**2
        t15 = 2*t13
        t16 = CDy*t3*t6
        t17 = ABy*beta
        t18 = t12*t15
        return np.pi**2.5*(2*F0*t5**(-0.5)*t8*t9/(p**3*q**2) + F3*q*t10*t14*t5**(-3.5) - 2*t10*t16*t17 - t11*t14*t4 - t12 + t15*t16*t4 - t15*t7*t9 + t17*t18 - t18*t8 + t4*t7)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 185:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABx*CDz
        t9 = alpha*gamma*t8
        t10 = t7*t9
        t11 = CDy*delta
        t12 = ABy*beta
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t1*t14
        t16 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t17 = ABx*t16
        t18 = t17*t3
        t19 = t15*t18
        t20 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t21 = CDz*t20
        t22 = t0*t14
        t23 = t21*t22
        t24 = t23*t4
        t25 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t26 = 2*t25**2
        t27 = alpha*t17
        t28 = t13*t7
        t29 = 2*t25
        t30 = t12*t29
        t31 = t11*t29
        t32 = gamma*t21
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t29*t35
        return np.pi**2.5*(-ABy*t14*t18*t2*t29 - CDy*t23*t29*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) - t10*t30 + t10*t31 + t10 - t11*t37 + t12*t37 - t13*t22*t3*t33 - t15*t26*t4*t8 + t19*t31 + t19 + t24*t30 - t24 + t27*t28 - t27*t36 - t28*t32 + t32*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 186:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = ABx*alpha
        t7 = q**(-2.0)
        t8 = p + q
        t9 = F1*t8**(-1.5)/p**2
        t10 = t7*t9
        t11 = t10*t6
        t12 = ABy*beta
        t13 = CDy*t11
        t14 = F0*t12*t6*t8**(-0.5)/p**3
        t15 = 2*CDy**2
        t16 = delta*t15
        t17 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t18 = t17**3
        t19 = F3*t8**(-3.5)
        t20 = 2*t19
        t21 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t22 = ABy*t21
        t23 = beta*t22
        t24 = t3*t9
        t25 = F2*t8**(-2.5)
        t26 = t0*t25
        t27 = CDy*t21
        t28 = t26*t27
        t29 = t17*t6
        t30 = t25*t3
        t31 = t1*t30
        t32 = gamma*t3
        t33 = t17**2
        t34 = 2*t33
        t35 = CDy*t26*t34*t6
        t36 = 2*CDy
        t37 = t12*t29*t36
        t38 = gamma*t10
        t39 = t16*t38
        t40 = t17*t21
        t41 = t26*t40
        t42 = t20*t33
        t43 = t27*t42
        t44 = ABy*t36*t40
        return np.pi**2.5*(-ABx*ABy*t2*t30*t34 + 2*F4*p*q*t18*t21*t8**(-4.5) + delta*t10*t37 - delta*t13 - delta*t43 + gamma*t13 + gamma*t31*t44 + gamma*t43 + gamma*t14*t16/q**3 - t1*t25*t4*t44 + t11*t12 - t14*t7 - t15*t41*t5 - t18*t20*t6 - 3*t19*t40 - t22*t31 + t23*t24 - t23*t39 + t23*t42 - t24*t29 + 3*t26*t29*t3 - t28*t32 + t28*t4 + t29*t39 - t32*t35 + t35*t4 - t37*t38 + t41)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 187:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F2*t6**(-2.5)
        t8 = t3*t7
        t9 = t0*t8
        t10 = CDx*gamma
        t11 = ABx*t10
        t12 = alpha*t11
        t13 = F1*t6**(-1.5)/(p**2*q**2)
        t14 = t12*t13
        t15 = ABy*beta
        t16 = CDy*t15
        t17 = delta*t16
        t18 = t13*t17
        t19 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t20 = t19**2
        t21 = F3*t6**(-3.5)
        t22 = t20*t21
        t23 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t24 = t23**2
        t25 = t21*t24
        t26 = ABx*t19
        t27 = t1*t8
        t28 = CDy*t23
        t29 = t0*t4*t7
        t30 = t15*t23
        t31 = t30*t9
        t32 = t10*t19
        t33 = 2*t24
        t34 = 2*t14
        t35 = alpha*t26
        t36 = delta*t28
        t37 = 2*t32
        t38 = 2*t25
        t39 = 2*t22
        t40 = 2*t26
        t41 = t28*t7
        return np.pi**2.5*(-ABy*t2*t23*t40*t8 - 2*CDx*t0*t19*t41*t5 + 2*F0*t12*t17*t6**(-0.5)/(p**3*q**3) + F4*p*q*t20*t33*t6**(-4.5) + t1*t4*t40*t41 - t11*t27*t33 + t14 - 2*t16*t20*t29 + 2*t18*t35 - t18*t37 + t18 - t22 - t25 + t26*t27 + t28*t29 - t30*t34 + t30*t39 + t31*t37 - t31 + t32*t38 - t32*t9 + t34*t36 - t35*t38 - t36*t39 + (1/2)*t9)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 188:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = ABy*beta
        t5 = p**(-2.0)
        t6 = p + q
        t7 = t6**(-1.5)
        t8 = F1*t5*t7
        t9 = t2*t8
        t10 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t11 = t6**(-2.5)
        t12 = F2*t0*t10*t11
        t13 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t14 = t13**2
        t15 = 2*t13
        t16 = t12*t15
        return np.pi**2.5*(2*ABx*ABy*CDx*F0*alpha*beta*delta*t6**(-0.5)/(p**3*q**2) - ABx*ABy*t1*t15*t9 + 2*ABx*CDx*F1*alpha*delta*t10*t2*t5*t7 - ABx*alpha*t16 + 2*ABy*F2*beta*t0*t11*t14 - CDx*delta*t16 - CDx*t15*t3*t4*t8 + 2*F3*q*t10*t14*t6**(-3.5) - t12 - t4*t9)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 189:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p**(-2.0)
        t7 = q**(-2.0)
        t8 = p + q
        t9 = t8**(-1.5)
        t10 = ABy*CDz*F1*beta*gamma*t6*t7*t9
        t11 = t8**(-2.5)
        t12 = F2*t11
        t13 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t14 = ABy*t13*t3
        t15 = t1*t12*t14
        t16 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t17 = CDz*t0*t12*t16
        t18 = t17*t4
        t19 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t20 = t19**2
        t21 = 2*t19
        t22 = ABx*alpha
        t23 = t21*t22
        t24 = CDx*delta
        t25 = t21*t24
        t26 = t8**(-3.5)
        t27 = F3*t13*t16*t26
        t28 = t21*t27
        return np.pi**2.5*(2*ABx*ABy*CDx*CDz*F0*alpha*beta*delta*gamma*t8**(-0.5)/(p**3*q**3) + 2*ABx*ABy*CDx*F1*alpha*beta*delta*t13*t6*t7*t9 + 2*ABx*CDx*CDz*F1*alpha*delta*gamma*t16*t6*t7*t9 + 2*ABx*CDx*F2*alpha*delta*t0*t11*t13*t16*t3 - ABx*t12*t14*t2*t21 + 2*ABy*CDz*F2*beta*gamma*t0*t11*t20*t3 + 2*ABy*F3*beta*t13*t20*t26 - CDx*t17*t21*t5 + 2*CDz*F3*gamma*t16*t20*t26 + 2*F4*p*q*t13*t16*t20*t8**(-4.5) - t10*t23 - t10*t25 - t10 - t15*t25 - t15 - t18*t23 - t18 - t22*t28 - t24*t28 - t27)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 190:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = t6**(-2.5)
        t8 = p**(-2.0)
        t9 = q**(-2.0)
        t10 = t6**(-1.5)
        t11 = F1*t10*t8*t9
        t12 = ABy*beta
        t13 = CDy*gamma
        t14 = t11*t12*t13
        t15 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t16 = t15**2
        t17 = t6**(-3.5)
        t18 = F3*t17
        t19 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t20 = t19**2
        t21 = t18*t20
        t22 = F2*t7
        t23 = t22*t3
        t24 = t0*t19*t23
        t25 = ABx*t15
        t26 = alpha*t25
        t27 = 2*t14
        t28 = CDx*t15
        t29 = delta*t28
        t30 = 2*t21
        t31 = 2*t19
        t32 = t23*t25
        t33 = t0*t28
        return np.pi**2.5*(2*ABx*ABy*CDx*CDy*F0*alpha*beta*delta*gamma*t6**(-0.5)/(p**3*q**3) + 2*ABx*ABy*CDx*F1*alpha*beta*delta*t10*t19*t8*t9 + 2*ABx*CDx*CDy*F1*alpha*delta*gamma*t10*t19*t8*t9 + 2*ABx*CDx*F2*alpha*delta*t0*t20*t3*t7 - ABx*CDx*alpha*delta*t11 + ABx*F2*alpha*t0*t15*t3*t7 + 2*ABy*CDy*F2*beta*gamma*t0*t16*t3*t7 + 2*ABy*F3*beta*t16*t17*t19 - ABy*t2*t31*t32 + CDx*F2*delta*t0*t15*t3*t7 + 2*CDy*F3*gamma*t16*t17*t19 - CDy*t22*t31*t33*t5 + (1/2)*F2*t0*t3*t7 + 2*F4*p*q*t16*t20*t6**(-4.5) - t1*t13*t31*t32 - t12*t22*t31*t33*t4 - t12*t24 - t13*t24 - t14 - t16*t18 - t21 - t26*t27 - t26*t30 - t27*t29 - t29*t30)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 191:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = q**(-2.0)
        t7 = ABy*beta
        t8 = p + q
        t9 = F1*t8**(-1.5)/p**2
        t10 = t6*t7*t9
        t11 = ABx*alpha
        t12 = t10*t11
        t13 = CDx*t10
        t14 = F0*t11*t7*t8**(-0.5)/p**3
        t15 = 2*CDx**2
        t16 = gamma*t15
        t17 = delta*t16
        t18 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t19 = t18**3
        t20 = F3*t8**(-3.5)
        t21 = 2*t20
        t22 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t23 = ABx*t22
        t24 = F2*t8**(-2.5)
        t25 = t24*t3
        t26 = t1*t25
        t27 = t3*t7
        t28 = t18*t27
        t29 = t0*t24
        t30 = CDx*t22
        t31 = t29*t30
        t32 = alpha*t23
        t33 = t32*t9
        t34 = t18**2
        t35 = 2*t34
        t36 = CDx*t4
        t37 = t29*t35
        t38 = CDx*gamma
        t39 = delta*t18
        t40 = 2*t38
        t41 = t18*t22
        t42 = t29*t41
        t43 = t21*t34
        t44 = t30*t43
        t45 = ABx*t41
        return np.pi**2.5*(-ABx*ABy*t2*t25*t35 + 2*CDx*t12*t39 + 2*F4*p*q*t19*t22*t8**(-4.5) + delta*t13 - delta*t44 - gamma*t13 - gamma*t3*t31 + gamma*t44 + 2*t1*t24*t36*t45 - t10*t16*t39 - t12*t18*t40 + t12 - t14*t6 - t15*t42*t5 + t17*t33*t6 + t19*t21*t7 - 3*t20*t41 + t23*t26 - t26*t40*t45 + t27*t37*t38 - 3*t28*t29 + t28*t9 - t3*t33 + t31*t4 - t32*t43 - t36*t37*t7 + t42 + t14*t17/q**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 192:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = p + q
        return -2*np.pi**2.5*t0*(ABx*F0*t1*t2*t3**(-0.5) + F1*t3**(-1.5)*(t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)))*np.exp(-alpha*rAB*t1 - delta*gamma*rCD*t2)
    if case_id == 193:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        return -2*np.pi**2.5*(ABx*CDz*F0*beta*gamma*t4**(-0.5)/(p**2*q**2) + ABx*t1*t2*t5*t6 + CDz*t0*t3*t6*t7 + F2*t4**(-2.5)*t5*t7)*np.exp(-alpha*rAB*t1 - delta*rCD*t3)
    if case_id == 194:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        return -2*np.pi**2.5*(ABx*CDy*F0*beta*gamma*t4**(-0.5)/(p**2*q**2) + ABx*t1*t2*t5*t6 + CDy*t0*t3*t6*t7 + F2*t4**(-2.5)*t5*t7)*np.exp(-alpha*rAB*t1 - delta*rCD*t3)
    if case_id == 195:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = t2**(-1.5)
        t4 = ABx*beta
        t5 = CDx*gamma
        t6 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        t7 = 2*F1*t0*t1*t3*t6
        return np.pi**2.5*(-2*F0*t2**(-0.5)*t4*t5/(p**2*q**2) + F1*t0*t1*t3 - 2*F2*t2**(-2.5)*t6**2 - t4*t7 - t5*t7)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 196:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        return 2*np.pi**2.5*(ABx*CDz*F0*beta*delta*t4**(-0.5)/(p**2*q**2) - ABx*t1*t2*t6*t7 + CDz*t0*t3*t5*t6 - F2*t4**(-2.5)*t5*t7)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 197:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = q**(-2.0)
        t7 = ABx*t6
        t8 = t1*t5*t7
        t9 = F0*beta*t4**(-0.5)/p**2
        t10 = 2*CDz**2*t3
        t11 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t12 = t11*t2
        t13 = F2*t4**(-2.5)
        t14 = t12*t13
        t15 = t0*t5
        t16 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t17 = 2*t16**2
        t18 = 2*CDz*t16
        t19 = t18*t8
        t20 = t14*t18
        return np.pi**2.5*(-ABx*beta*t13*t17*t2 + ABx*t10*t9/q**3 - F3*p*t11*t17*t4**(-3.5) + delta*t19 + delta*t20 - gamma*t19 - gamma*t20 + t10*t11*t15*t6 - t12*t15 + t14 - t7*t9 + t8)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 198:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = CDz*delta
        t6 = CDy*gamma
        t7 = ABx*t6
        t8 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t9 = ABx*t8
        t10 = q**(-2.0)
        t11 = t4**(-1.5)
        t12 = F1*t1*t10*t11
        t13 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t14 = t13*t6
        t15 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t16 = t13*t8
        t17 = F2*t4**(-2.5)
        t18 = t15*t17*t2
        return 2*np.pi**2.5*(CDz*t16*t17*t3 + F0*beta*t4**(-0.5)*t5*t7/(p**2*q**3) + F1*t0*t10*t11*t14*t5 - F3*p*t15*t16*t4**(-3.5) - beta*t18*t9 - t12*t15*t7 + t12*t5*t9 - t14*t18)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 199:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDz*delta
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABx
        t9 = beta*t8
        t10 = CDx*CDz*t3
        t11 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t15 = 2*t14**2
        t16 = CDx*gamma
        t17 = t1*t8
        t18 = t13*t14
        return np.pi**2.5*(F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 - t0*t7 - t11*t16*t17*t6 + t12*t15*t4 + t13 + t14*t17*t7 - 2*t16*t18 - t18*t9)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 200:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t6 = F1*t4**(-1.5)
        t7 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        return 2*np.pi**2.5*(ABx*CDy*F0*beta*delta*t4**(-0.5)/(p**2*q**2) - ABx*t1*t2*t6*t7 + CDy*t0*t3*t5*t6 - F2*t4**(-2.5)*t5*t7)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 201:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = CDy*delta
        t6 = CDz*gamma
        t7 = ABx*t6
        t8 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t9 = ABx*t8
        t10 = q**(-2.0)
        t11 = t4**(-1.5)
        t12 = F1*t1*t10*t11
        t13 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t14 = t13*t6
        t15 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t16 = t13*t8
        t17 = F2*t4**(-2.5)
        t18 = t15*t17*t2
        return 2*np.pi**2.5*(CDy*t16*t17*t3 + F0*beta*t4**(-0.5)*t5*t7/(p**2*q**3) + F1*t0*t10*t11*t14*t5 - F3*p*t15*t16*t4**(-3.5) - beta*t18*t9 - t12*t15*t7 + t12*t5*t9 - t14*t18)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 202:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = q**(-2.0)
        t7 = ABx*t6
        t8 = t1*t5*t7
        t9 = F0*beta*t4**(-0.5)/p**2
        t10 = 2*CDy**2*t3
        t11 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t12 = t11*t2
        t13 = F2*t4**(-2.5)
        t14 = t12*t13
        t15 = t0*t5
        t16 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t17 = 2*t16**2
        t18 = 2*CDy*t16
        t19 = t18*t8
        t20 = t14*t18
        return np.pi**2.5*(-ABx*beta*t13*t17*t2 + ABx*t10*t9/q**3 - F3*p*t11*t17*t4**(-3.5) + delta*t19 + delta*t20 - gamma*t19 - gamma*t20 + t10*t11*t15*t6 - t12*t15 + t14 - t7*t9 + t8)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 203:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDy*delta
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABx
        t9 = beta*t8
        t10 = CDx*CDy*t3
        t11 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t15 = 2*t14**2
        t16 = CDx*gamma
        t17 = t1*t8
        t18 = t13*t14
        return np.pi**2.5*(F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 - t0*t7 - t11*t16*t17*t6 + t12*t15*t4 + t13 + t14*t17*t7 - 2*t16*t18 - t18*t9)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 204:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = F1*t0*t1*t2**(-1.5)
        t4 = ABx*beta
        t5 = CDx*delta
        t6 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        t7 = 2*t3*t6
        return np.pi**2.5*(2*F0*t2**(-0.5)*t4*t5/(p**2*q**2) - 2*F2*t2**(-2.5)*t6**2 + t3 - t4*t7 + t5*t7)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 205:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDz*gamma
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABx
        t9 = beta*t8
        t10 = CDx*CDz*t3
        t11 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t15 = 2*t14**2
        t16 = CDx*delta
        t17 = t1*t8
        t18 = t13*t14
        return np.pi**2.5*(F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 + t0*t7 + t11*t16*t17*t6 - t12*t15*t4 + t13 - t14*t17*t7 + 2*t16*t18 - t18*t9)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 206:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = CDy*gamma
        t5 = p + q
        t6 = F1*t5**(-1.5)/q**2
        t7 = t4*t6
        t8 = 2*ABx
        t9 = beta*t8
        t10 = CDx*CDy*t3
        t11 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t12 = F2*t2*t5**(-2.5)
        t13 = t11*t12
        t14 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t15 = 2*t14**2
        t16 = CDx*delta
        t17 = t1*t8
        t18 = t13*t14
        return np.pi**2.5*(F0*t10*t5**(-0.5)*t9/(p**2*q**3) - F3*p*t11*t15*t5**(-3.5) + 2*t0*t10*t14*t6 + t0*t7 + t11*t16*t17*t6 - t12*t15*t4 + t13 - t14*t17*t7 + 2*t16*t18 - t18*t9)*np.exp(-alpha*rAB*t1 - rCD*t2*t3)
    if case_id == 207:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = q**(-2.0)
        t7 = ABx*t6
        t8 = t1*t5*t7
        t9 = CDx*gamma
        t10 = t0*t5
        t11 = t10*t6
        t12 = F0*beta*t4**(-0.5)/p**2
        t13 = CDx*delta
        t14 = 2*ABx
        t15 = CDx**2*delta*gamma
        t16 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t17 = t16*t2
        t18 = F2*t4**(-2.5)
        t19 = t16**2*t18
        t20 = t19*t2
        t21 = 2*t13
        t22 = 2*t16
        return np.pi**2.5*(-2*CDx*t19*t3 - 2*F3*p*t16**3*t4**(-3.5) - beta*t14*t20 - t10*t17 - t11*t13 + t11*t15*t22 + t11*t9 - t12*t7 + t16*t21*t8 + 3*t17*t18 + t20*t21 - t22*t8*t9 + t8 + t12*t14*t15/q**3)*np.exp(-alpha*rAB*t1 - delta*rCD*t3)
    if case_id == 208:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = ABz*alpha
        t4 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        t5 = t2**(-1.5)
        t6 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        return 2*np.pi**2.5*t0*(-ABx*F0*beta*t1*t2**(-0.5)*t3/p**2 + ABx*F1*beta*t0*t5*t6 - F1*t0*t3*t4*t5 + F2*q*t2**(-2.5)*t4*t6)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 209:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = F1*t4*t6
        t8 = t2*t7
        t9 = ABx*ABz*t1
        t10 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t11 = t5**(-2.5)
        t12 = F2*t0*t10*t11
        t13 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t14 = t13**2
        t15 = 2*t13
        t16 = ABz*alpha
        return np.pi**2.5*(2*ABx*CDz*F1*beta*gamma*t13*t2*t4*t6 + 2*ABx*F2*beta*t0*t11*t14 - ABx*beta*t8 - 2*CDz*F0*gamma*t5**(-0.5)*t9/(p**3*q**2) + 2*CDz*F2*gamma*t0*t10*t11*t13 - 2*CDz*t10*t16*t3*t7 + 2*F3*q*t10*t14*t5**(-3.5) - t12*t15*t16 - t12 - t15*t8*t9)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 210:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = gamma*t1
        t3 = p + q
        t4 = ABx*beta
        t5 = ABz*alpha
        t6 = CDy*t5
        t7 = p**(-2.0)
        t8 = t3**(-1.5)
        t9 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        t10 = t5*t9
        t11 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        t12 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        t13 = t3**(-2.5)
        return 2*np.pi**2.5*(ABx*CDy*F1*beta*gamma*t1*t12*t7*t8 + ABx*F2*beta*t0*t12*t13*t9 + CDy*F2*gamma*t0*t11*t12*t13 - F0*gamma*t3**(-0.5)*t4*t6/(p**3*q**2) - F1*t1*t10*t4*t7*t8 - F1*t11*t2*t6*t7*t8 - F2*t0*t10*t11*t13 + F3*q*t11*t12*t3**(-3.5)*t9)*np.exp(-alpha*beta*rAB*t0 - delta*rCD*t2)
    if case_id == 211:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = ABx*ABz*t1
        t8 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t9 = t5**(-2.5)
        t10 = F2*t0*t9
        t11 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t12 = t11**2
        t13 = ABz*alpha
        t14 = F1*t4*t6
        t15 = 2*t11
        return np.pi**2.5*(2*ABx*CDx*F1*beta*gamma*t2*t4*t6*t8 + 2*ABx*F2*beta*t0*t11*t8*t9 + ABz*F1*alpha*t2*t4*t6 - 2*CDx*F0*gamma*t5**(-0.5)*t7/(p**3*q**2) + 2*CDx*F2*gamma*t0*t11*t8*t9 - CDx*t13*t14*t15*t3 + 2*F3*q*t12*t5**(-3.5)*t8 - 2*t10*t12*t13 - t10*t8 - t14*t15*t2*t7)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 212:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = ABx*beta
        t5 = p**(-2.0)
        t6 = p + q
        t7 = t6**(-1.5)
        t8 = F1*t5*t7
        t9 = t2*t8
        t10 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t11 = t6**(-2.5)
        t12 = F2*t0*t10*t11
        t13 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t14 = t13**2
        t15 = 2*t13
        t16 = t12*t15
        return np.pi**2.5*(2*ABx*ABz*CDz*F0*alpha*beta*delta*t6**(-0.5)/(p**3*q**2) - ABx*ABz*t1*t15*t9 + 2*ABx*F2*beta*t0*t11*t14 + 2*ABz*CDz*F1*alpha*delta*t10*t2*t5*t7 - ABz*alpha*t16 - CDz*delta*t16 - CDz*t15*t3*t4*t8 + 2*F3*q*t10*t14*t6**(-3.5) - t12 - t4*t9)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 213:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = q**(-2.0)
        t7 = ABx*beta
        t8 = p + q
        t9 = F1*t8**(-1.5)/p**2
        t10 = t6*t7*t9
        t11 = ABz*alpha
        t12 = t10*t11
        t13 = CDz*t10
        t14 = F0*t11*t7*t8**(-0.5)/p**3
        t15 = 2*CDz**2
        t16 = gamma*t15
        t17 = delta*t16
        t18 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t19 = t18**3
        t20 = F3*t8**(-3.5)
        t21 = 2*t20
        t22 = t3*t7
        t23 = t18*t22
        t24 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t25 = ABz*t24
        t26 = F2*t8**(-2.5)
        t27 = t26*t3
        t28 = t1*t27
        t29 = t0*t26
        t30 = CDz*t24
        t31 = t29*t30
        t32 = alpha*t25
        t33 = t32*t9
        t34 = t18**2
        t35 = 2*t34
        t36 = CDz*t4
        t37 = t29*t35
        t38 = CDz*gamma
        t39 = delta*t18
        t40 = 2*t38
        t41 = t18*t24
        t42 = t29*t41
        t43 = t21*t34
        t44 = t30*t43
        t45 = ABz*t41
        return np.pi**2.5*(-ABx*ABz*t2*t27*t35 + 2*CDz*t12*t39 + 2*F4*p*q*t19*t24*t8**(-4.5) + delta*t13 - delta*t44 - gamma*t13 - gamma*t3*t31 + gamma*t44 + 2*t1*t26*t36*t45 - t10*t16*t39 - t12*t18*t40 + t12 - t14*t6 - t15*t42*t5 + t17*t33*t6 + t19*t21*t7 - 3*t20*t41 + t22*t37*t38 - 3*t23*t29 + t23*t9 + t25*t28 - t28*t40*t45 - t3*t33 + t31*t4 - t32*t43 - t36*t37*t7 + t42 + t14*t17/q**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 214:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p**(-2.0)
        t7 = q**(-2.0)
        t8 = p + q
        t9 = t8**(-1.5)
        t10 = ABx*CDy*F1*beta*gamma*t6*t7*t9
        t11 = t8**(-2.5)
        t12 = F2*t11
        t13 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t14 = ABx*t13*t3
        t15 = t1*t12*t14
        t16 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t17 = CDy*t0*t12*t16
        t18 = t17*t4
        t19 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t20 = t19**2
        t21 = 2*t19
        t22 = ABz*alpha
        t23 = t21*t22
        t24 = CDz*delta
        t25 = t21*t24
        t26 = t8**(-3.5)
        t27 = F3*t13*t16*t26
        t28 = t21*t27
        return np.pi**2.5*(2*ABx*ABz*CDy*CDz*F0*alpha*beta*delta*gamma*t8**(-0.5)/(p**3*q**3) + 2*ABx*ABz*CDz*F1*alpha*beta*delta*t13*t6*t7*t9 + 2*ABx*CDy*F2*beta*gamma*t0*t11*t20*t3 + 2*ABx*F3*beta*t13*t20*t26 + 2*ABz*CDy*CDz*F1*alpha*delta*gamma*t16*t6*t7*t9 + 2*ABz*CDz*F2*alpha*delta*t0*t11*t13*t16*t3 - ABz*t12*t14*t2*t21 + 2*CDy*F3*gamma*t16*t20*t26 - CDz*t17*t21*t5 + 2*F4*p*q*t13*t16*t20*t8**(-4.5) - t10*t23 - t10*t25 - t10 - t15*t25 - t15 - t18*t23 - t18 - t22*t28 - t24*t28 - t27)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 215:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = t6**(-2.5)
        t8 = ABx*beta
        t9 = CDx*gamma
        t10 = p**(-2.0)
        t11 = q**(-2.0)
        t12 = t6**(-1.5)
        t13 = F1*t10*t11*t12
        t14 = t13*t8*t9
        t15 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t16 = t15**2
        t17 = t6**(-3.5)
        t18 = F3*t17
        t19 = t16*t18
        t20 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t21 = t20**2
        t22 = F2*t7
        t23 = t22*t3
        t24 = t0*t15*t23
        t25 = ABz*t20
        t26 = alpha*t25
        t27 = 2*t14
        t28 = CDz*t20
        t29 = delta*t28
        t30 = 2*t19
        t31 = 2*t15
        t32 = t23*t25
        t33 = t0*t28
        return np.pi**2.5*(2*ABx*ABz*CDx*CDz*F0*alpha*beta*delta*gamma*t6**(-0.5)/(p**3*q**3) + 2*ABx*ABz*CDz*F1*alpha*beta*delta*t10*t11*t12*t15 + 2*ABx*CDx*F2*beta*gamma*t0*t21*t3*t7 + 2*ABx*F3*beta*t15*t17*t21 - ABx*t2*t31*t32 + 2*ABz*CDx*CDz*F1*alpha*delta*gamma*t10*t11*t12*t15 + 2*ABz*CDz*F2*alpha*delta*t0*t16*t3*t7 - ABz*CDz*alpha*delta*t13 + ABz*F2*alpha*t0*t20*t3*t7 + 2*CDx*F3*gamma*t15*t17*t21 - CDx*t22*t31*t33*t5 + CDz*F2*delta*t0*t20*t3*t7 + (1/2)*F2*t0*t3*t7 + 2*F4*p*q*t16*t21*t6**(-4.5) - t1*t31*t32*t9 - t14 - t18*t21 - t19 - t22*t31*t33*t4*t8 - t24*t8 - t24*t9 - t26*t27 - t26*t30 - t27*t29 - t29*t30)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 216:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = ABz*alpha
        t6 = CDy*t5
        t7 = ABx*beta
        t8 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t9 = F1*t4**(-1.5)/p**2
        t10 = t3*t9
        t11 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t12 = t11*t5
        t13 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t14 = CDy*t13
        t15 = t11*t13
        t16 = F2*t4**(-2.5)
        t17 = t0*t16*t8
        return 2*np.pi**2.5*(ABx*t1*t15*t16 + F0*delta*t4**(-0.5)*t6*t7/(p**3*q**2) + F3*q*t15*t4**(-3.5)*t8 - delta*t14*t17 - t10*t14*t7 + t10*t6*t8 - t12*t17 - t12*t2*t7*t9)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 217:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABx*CDy
        t9 = beta*delta*t8
        t10 = t7*t9
        t11 = ABz*alpha
        t12 = CDz*gamma
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t14*t4
        t16 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t17 = CDy*t16
        t18 = t0*t17
        t19 = t15*t18
        t20 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t21 = ABx*t20
        t22 = t14*t3
        t23 = t21*t22
        t24 = t1*t23
        t25 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t26 = 2*t25**2
        t27 = 2*t25
        t28 = t11*t27
        t29 = beta*t21
        t30 = t13*t7
        t31 = t12*t27
        t32 = delta*t17
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t27*t35
        return np.pi**2.5*(-ABz*t2*t23*t27 - CDz*t14*t18*t27*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) - t0*t13*t22*t33 - t1*t15*t26*t8 + t10*t28 - t10*t31 + t10 - t11*t37 + t12*t37 + t19*t28 + t19 + t24*t31 - t24 - t29*t30 + t29*t36 + t30*t32 - t32*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 218:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = q**(-2.0)
        t7 = p + q
        t8 = F1*t7**(-1.5)/p**2
        t9 = t6*t8
        t10 = ABx*beta
        t11 = ABz*alpha*t10
        t12 = t11*t9
        t13 = F0*t11*t7**(-0.5)/p**3
        t14 = CDy**2
        t15 = 2*delta
        t16 = gamma*t14*t15
        t17 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t18 = t10*t17
        t19 = t18*t3
        t20 = F2*t7**(-2.5)
        t21 = t20*t3
        t22 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t23 = ABz*t22
        t24 = t1*t23
        t25 = t21*t24
        t26 = t0*t20
        t27 = t19*t26
        t28 = alpha*t23
        t29 = t28*t8
        t30 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t31 = 2*t30**2
        t32 = CDy*t30
        t33 = 2*gamma
        t34 = t32*t33
        t35 = t17*t22
        t36 = F3*t7**(-3.5)
        t37 = t35*t36
        t38 = t26*t35
        t39 = t31*t36
        t40 = 2*t32*t4
        t41 = t32*t37
        return np.pi**2.5*(-ABx*ABz*t2*t21*t31 + F4*p*q*t31*t35*t7**(-4.5) + t12*t15*t32 - t12*t34 + t12 - t13*t6 - 2*t14*t38*t5 - t15*t41 - t16*t18*t9 + t16*t29*t6 - t18*t26*t40 + t18*t39 + t19*t8 + t20*t24*t40 - t25*t34 + t25 + t27*t34 - t27 - t28*t39 - t29*t3 + t33*t41 - t37 + t38 + t13*t16/q**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 219:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABz*CDy
        t9 = alpha*delta*t8
        t10 = t7*t9
        t11 = ABx*beta
        t12 = CDx*gamma
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t1*t14
        t16 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t17 = ABz*t16
        t18 = t17*t3
        t19 = t15*t18
        t20 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t21 = CDy*t20
        t22 = t0*t14
        t23 = t21*t22
        t24 = t23*t4
        t25 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t26 = 2*t25**2
        t27 = alpha*t17
        t28 = t13*t7
        t29 = 2*t25
        t30 = t11*t29
        t31 = delta*t21
        t32 = t12*t29
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t29*t35
        return np.pi**2.5*(-ABx*t14*t18*t2*t29 - CDx*t23*t29*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) + t10*t30 + t10*t32 - t10 + t11*t37 + t12*t37 + t13*t22*t3*t33 + t15*t26*t4*t8 - t19*t32 + t19 - t24*t30 + t24 - t27*t28 - t27*t36 - t28*t31 - t31*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 220:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = ABz*alpha
        t5 = p + q
        t6 = F1*t5**(-1.5)/p**2
        t7 = t2*t6
        t8 = CDx*delta
        t9 = ABx*ABz*t1
        t10 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t11 = F2*t0*t5**(-2.5)
        t12 = t10*t11
        t13 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t14 = 2*t13**2
        t15 = 2*t13
        t16 = ABx*beta
        t17 = CDx*t3*t6
        t18 = t12*t15
        return np.pi**2.5*(2*F0*t5**(-0.5)*t8*t9/(p**3*q**2) + F3*q*t10*t14*t5**(-3.5) - 2*t10*t16*t17 - t11*t14*t4 - t12 + t15*t17*t4 - t15*t7*t9 + t16*t18 - t18*t8 + t4*t7)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 221:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F2*t6**(-2.5)
        t8 = t3*t7
        t9 = t0*t8
        t10 = ABx*beta
        t11 = CDx*t10
        t12 = delta*t11
        t13 = F1*t6**(-1.5)/(p**2*q**2)
        t14 = t12*t13
        t15 = CDz*gamma
        t16 = ABz*t15
        t17 = alpha*t16
        t18 = t13*t17
        t19 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t20 = t19**2
        t21 = F3*t6**(-3.5)
        t22 = t20*t21
        t23 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t24 = t23**2
        t25 = t21*t24
        t26 = ABz*t23
        t27 = t1*t8
        t28 = CDx*t19
        t29 = t0*t4*t7
        t30 = t10*t19
        t31 = t30*t9
        t32 = t15*t23
        t33 = 2*t24
        t34 = alpha*t26
        t35 = 2*t18
        t36 = 2*t32
        t37 = delta*t28
        t38 = 2*t25
        t39 = 2*t22
        t40 = 2*t26
        t41 = t28*t7
        return np.pi**2.5*(-ABx*t19*t2*t40*t8 - 2*CDz*t0*t23*t41*t5 + 2*F0*t12*t17*t6**(-0.5)/(p**3*q**3) + F4*p*q*t20*t33*t6**(-4.5) + t1*t4*t40*t41 - t11*t29*t33 + 2*t14*t34 - t14*t36 + t14 - 2*t16*t20*t27 + t18 - t22 - t25 + t26*t27 + t28*t29 - t30*t35 + t30*t38 + t31*t36 - t31 + t32*t39 - t32*t9 - t34*t39 + t35*t37 - t37*t38 + (1/2)*t9)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 222:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABz*CDy
        t9 = alpha*gamma*t8
        t10 = t7*t9
        t11 = CDx*delta
        t12 = ABx*beta
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t1*t14
        t16 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t17 = ABz*t16
        t18 = t17*t3
        t19 = t15*t18
        t20 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t21 = CDy*t20
        t22 = t0*t14
        t23 = t21*t22
        t24 = t23*t4
        t25 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t26 = 2*t25**2
        t27 = alpha*t17
        t28 = t13*t7
        t29 = 2*t25
        t30 = t12*t29
        t31 = gamma*t21
        t32 = t11*t29
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t29*t35
        return np.pi**2.5*(-ABx*t14*t18*t2*t29 - CDx*t23*t29*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) - t10*t30 + t10*t32 + t10 - t11*t37 + t12*t37 - t13*t22*t3*t33 - t15*t26*t4*t8 + t19*t32 + t19 + t24*t30 - t24 + t27*t28 - t27*t36 - t28*t31 + t31*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 223:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = ABz*alpha
        t7 = q**(-2.0)
        t8 = p + q
        t9 = F1*t8**(-1.5)/p**2
        t10 = t7*t9
        t11 = t10*t6
        t12 = ABx*beta
        t13 = CDx*t11
        t14 = F0*t12*t6*t8**(-0.5)/p**3
        t15 = 2*CDx**2
        t16 = delta*t15
        t17 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t18 = t17**3
        t19 = F3*t8**(-3.5)
        t20 = 2*t19
        t21 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t22 = ABx*t21
        t23 = beta*t22
        t24 = t3*t9
        t25 = F2*t8**(-2.5)
        t26 = t0*t25
        t27 = CDx*t21
        t28 = t26*t27
        t29 = t25*t3
        t30 = t1*t29
        t31 = t17*t6
        t32 = gamma*t3
        t33 = t17**2
        t34 = 2*t33
        t35 = CDx*t26*t34*t6
        t36 = 2*CDx
        t37 = t12*t31*t36
        t38 = gamma*t10
        t39 = t16*t38
        t40 = t17*t21
        t41 = t26*t40
        t42 = t20*t33
        t43 = t27*t42
        t44 = ABx*t36*t40
        return np.pi**2.5*(-ABx*ABz*t2*t29*t34 + 2*F4*p*q*t18*t21*t8**(-4.5) + delta*t10*t37 - delta*t13 - delta*t43 + gamma*t13 + gamma*t30*t44 + gamma*t43 + gamma*t14*t16/q**3 - t1*t25*t4*t44 + t11*t12 - t14*t7 - t15*t41*t5 - t18*t20*t6 - 3*t19*t40 - t22*t30 + t23*t24 - t23*t39 + t23*t42 - t24*t31 + 3*t26*t3*t31 - t28*t32 + t28*t4 + t31*t39 - t32*t35 + t35*t4 - t37*t38 + t41)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 224:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = p + q
        t3 = ABy*alpha
        t4 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        t5 = t2**(-1.5)
        t6 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        return 2*np.pi**2.5*t0*(-ABx*F0*beta*t1*t2**(-0.5)*t3/p**2 + ABx*F1*beta*t0*t5*t6 - F1*t0*t3*t4*t5 + F2*q*t2**(-2.5)*t4*t6)*np.exp(-alpha*beta*rAB*t0 - delta*gamma*rCD*t1)
    if case_id == 225:
        t0 = p**(-1.0)
        t1 = q**(-1.0)
        t2 = gamma*t1
        t3 = p + q
        t4 = ABx*beta
        t5 = ABy*alpha
        t6 = CDz*t5
        t7 = p**(-2.0)
        t8 = t3**(-1.5)
        t9 = t0*(Az*alpha + Bz*beta) - t1*(Cz*gamma + Dz*delta)
        t10 = t5*t9
        t11 = t0*(Ax*alpha + Bx*beta) - t1*(Cx*gamma + Dx*delta)
        t12 = t0*(Ay*alpha + By*beta) - t1*(Cy*gamma + Dy*delta)
        t13 = t3**(-2.5)
        return 2*np.pi**2.5*(ABx*CDz*F1*beta*gamma*t1*t12*t7*t8 + ABx*F2*beta*t0*t12*t13*t9 + CDz*F2*gamma*t0*t11*t12*t13 - F0*gamma*t3**(-0.5)*t4*t6/(p**3*q**2) - F1*t1*t10*t4*t7*t8 - F1*t11*t2*t6*t7*t8 - F2*t0*t10*t11*t13 + F3*q*t11*t12*t3**(-3.5)*t9)*np.exp(-alpha*beta*rAB*t0 - delta*rCD*t2)
    if case_id == 226:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = F1*t4*t6
        t8 = t2*t7
        t9 = ABx*ABy*t1
        t10 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t11 = t5**(-2.5)
        t12 = F2*t0*t10*t11
        t13 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t14 = t13**2
        t15 = 2*t13
        t16 = ABy*alpha
        return np.pi**2.5*(2*ABx*CDy*F1*beta*gamma*t13*t2*t4*t6 + 2*ABx*F2*beta*t0*t11*t14 - ABx*beta*t8 - 2*CDy*F0*gamma*t5**(-0.5)*t9/(p**3*q**2) + 2*CDy*F2*gamma*t0*t10*t11*t13 - 2*CDy*t10*t16*t3*t7 + 2*F3*q*t10*t14*t5**(-3.5) - t12*t15*t16 - t12 - t15*t8*t9)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 227:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p**(-2.0)
        t5 = p + q
        t6 = t5**(-1.5)
        t7 = ABx*ABy*t1
        t8 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t9 = t5**(-2.5)
        t10 = F2*t0*t9
        t11 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t12 = t11**2
        t13 = ABy*alpha
        t14 = F1*t4*t6
        t15 = 2*t11
        return np.pi**2.5*(2*ABx*CDx*F1*beta*gamma*t2*t4*t6*t8 + 2*ABx*F2*beta*t0*t11*t8*t9 + ABy*F1*alpha*t2*t4*t6 - 2*CDx*F0*gamma*t5**(-0.5)*t7/(p**3*q**2) + 2*CDx*F2*gamma*t0*t11*t8*t9 - CDx*t13*t14*t15*t3 + 2*F3*q*t12*t5**(-3.5)*t8 - 2*t10*t12*t13 - t10*t8 - t14*t15*t2*t7)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 228:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = ABy*alpha
        t6 = CDz*t5
        t7 = ABx*beta
        t8 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t9 = F1*t4**(-1.5)/p**2
        t10 = t3*t9
        t11 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t12 = t11*t5
        t13 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t14 = CDz*t13
        t15 = t11*t13
        t16 = F2*t4**(-2.5)
        t17 = t0*t16*t8
        return 2*np.pi**2.5*(ABx*t1*t15*t16 + F0*delta*t4**(-0.5)*t6*t7/(p**3*q**2) + F3*q*t15*t4**(-3.5)*t8 - delta*t14*t17 - t10*t14*t7 + t10*t6*t8 - t12*t17 - t12*t2*t7*t9)*np.exp(-alpha*rAB*t1 - gamma*rCD*t3)
    if case_id == 229:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = q**(-2.0)
        t7 = p + q
        t8 = F1*t7**(-1.5)/p**2
        t9 = t6*t8
        t10 = ABx*beta
        t11 = ABy*alpha*t10
        t12 = t11*t9
        t13 = F0*t11*t7**(-0.5)/p**3
        t14 = CDz**2
        t15 = 2*delta
        t16 = gamma*t14*t15
        t17 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t18 = t10*t17
        t19 = t18*t3
        t20 = F2*t7**(-2.5)
        t21 = t20*t3
        t22 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t23 = ABy*t22
        t24 = t1*t23
        t25 = t21*t24
        t26 = t0*t20
        t27 = t19*t26
        t28 = alpha*t23
        t29 = t28*t8
        t30 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t31 = 2*t30**2
        t32 = CDz*t30
        t33 = 2*gamma
        t34 = t32*t33
        t35 = t17*t22
        t36 = F3*t7**(-3.5)
        t37 = t35*t36
        t38 = t26*t35
        t39 = t31*t36
        t40 = 2*t32*t4
        t41 = t32*t37
        return np.pi**2.5*(-ABx*ABy*t2*t21*t31 + F4*p*q*t31*t35*t7**(-4.5) + t12*t15*t32 - t12*t34 + t12 - t13*t6 - 2*t14*t38*t5 - t15*t41 - t16*t18*t9 + t16*t29*t6 - t18*t26*t40 + t18*t39 + t19*t8 + t20*t24*t40 - t25*t34 + t25 + t27*t34 - t27 - t28*t39 - t29*t3 + t33*t41 - t37 + t38 + t13*t16/q**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 230:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABx*CDz
        t9 = beta*delta*t8
        t10 = t7*t9
        t11 = ABy*alpha
        t12 = CDy*gamma
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t14*t4
        t16 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t17 = CDz*t16
        t18 = t0*t17
        t19 = t15*t18
        t20 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t21 = ABx*t20
        t22 = t14*t3
        t23 = t21*t22
        t24 = t1*t23
        t25 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t26 = 2*t25**2
        t27 = beta*t21
        t28 = t13*t7
        t29 = 2*t25
        t30 = t11*t29
        t31 = t12*t29
        t32 = delta*t17
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t29*t35
        return np.pi**2.5*(-ABy*t2*t23*t29 - CDy*t14*t18*t29*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) - t0*t13*t22*t33 - t1*t15*t26*t8 + t10*t30 - t10*t31 + t10 - t11*t37 + t12*t37 + t19*t30 + t19 + t24*t31 - t24 - t27*t28 + t27*t36 + t28*t32 - t32*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 231:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABy*CDz
        t9 = alpha*delta*t8
        t10 = t7*t9
        t11 = ABx*beta
        t12 = CDx*gamma
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t1*t14
        t16 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t17 = ABy*t16
        t18 = t17*t3
        t19 = t15*t18
        t20 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t21 = CDz*t20
        t22 = t0*t14
        t23 = t21*t22
        t24 = t23*t4
        t25 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t26 = 2*t25**2
        t27 = alpha*t17
        t28 = t13*t7
        t29 = 2*t25
        t30 = t11*t29
        t31 = delta*t21
        t32 = t12*t29
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t29*t35
        return np.pi**2.5*(-ABx*t14*t18*t2*t29 - CDx*t23*t29*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) + t10*t30 + t10*t32 - t10 + t11*t37 + t12*t37 + t13*t22*t3*t33 + t15*t26*t4*t8 - t19*t32 + t19 - t24*t30 + t24 - t27*t28 - t27*t36 - t28*t31 - t31*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 232:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = ABx*beta
        t5 = p**(-2.0)
        t6 = p + q
        t7 = t6**(-1.5)
        t8 = F1*t5*t7
        t9 = t2*t8
        t10 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t11 = t6**(-2.5)
        t12 = F2*t0*t10*t11
        t13 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t14 = t13**2
        t15 = 2*t13
        t16 = t12*t15
        return np.pi**2.5*(2*ABx*ABy*CDy*F0*alpha*beta*delta*t6**(-0.5)/(p**3*q**2) - ABx*ABy*t1*t15*t9 + 2*ABx*F2*beta*t0*t11*t14 + 2*ABy*CDy*F1*alpha*delta*t10*t2*t5*t7 - ABy*alpha*t16 - CDy*delta*t16 - CDy*t15*t3*t4*t8 + 2*F3*q*t10*t14*t6**(-3.5) - t12 - t4*t9)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 233:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p**(-2.0)
        t7 = q**(-2.0)
        t8 = p + q
        t9 = t8**(-1.5)
        t10 = ABx*CDz*F1*beta*gamma*t6*t7*t9
        t11 = t8**(-2.5)
        t12 = F2*t11
        t13 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t14 = ABx*t13*t3
        t15 = t1*t12*t14
        t16 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t17 = CDz*t0*t12*t16
        t18 = t17*t4
        t19 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t20 = t19**2
        t21 = 2*t19
        t22 = ABy*alpha
        t23 = t21*t22
        t24 = CDy*delta
        t25 = t21*t24
        t26 = t8**(-3.5)
        t27 = F3*t13*t16*t26
        t28 = t21*t27
        return np.pi**2.5*(2*ABx*ABy*CDy*CDz*F0*alpha*beta*delta*gamma*t8**(-0.5)/(p**3*q**3) + 2*ABx*ABy*CDy*F1*alpha*beta*delta*t13*t6*t7*t9 + 2*ABx*CDz*F2*beta*gamma*t0*t11*t20*t3 + 2*ABx*F3*beta*t13*t20*t26 + 2*ABy*CDy*CDz*F1*alpha*delta*gamma*t16*t6*t7*t9 + 2*ABy*CDy*F2*alpha*delta*t0*t11*t13*t16*t3 - ABy*t12*t14*t2*t21 - CDy*t17*t21*t5 + 2*CDz*F3*gamma*t16*t20*t26 + 2*F4*p*q*t13*t16*t20*t8**(-4.5) - t10*t23 - t10*t25 - t10 - t15*t25 - t15 - t18*t23 - t18 - t22*t28 - t24*t28 - t27)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 234:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = q**(-2.0)
        t7 = ABx*beta
        t8 = p + q
        t9 = F1*t8**(-1.5)/p**2
        t10 = t6*t7*t9
        t11 = ABy*alpha
        t12 = t10*t11
        t13 = CDy*t10
        t14 = F0*t11*t7*t8**(-0.5)/p**3
        t15 = 2*CDy**2
        t16 = gamma*t15
        t17 = delta*t16
        t18 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t19 = t18**3
        t20 = F3*t8**(-3.5)
        t21 = 2*t20
        t22 = t3*t7
        t23 = t18*t22
        t24 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t25 = ABy*t24
        t26 = F2*t8**(-2.5)
        t27 = t26*t3
        t28 = t1*t27
        t29 = t0*t26
        t30 = CDy*t24
        t31 = t29*t30
        t32 = alpha*t25
        t33 = t32*t9
        t34 = t18**2
        t35 = 2*t34
        t36 = CDy*t4
        t37 = t29*t35
        t38 = CDy*gamma
        t39 = delta*t18
        t40 = 2*t38
        t41 = t18*t24
        t42 = t29*t41
        t43 = t21*t34
        t44 = t30*t43
        t45 = ABy*t41
        return np.pi**2.5*(-ABx*ABy*t2*t27*t35 + 2*CDy*t12*t39 + 2*F4*p*q*t19*t24*t8**(-4.5) + delta*t13 - delta*t44 - gamma*t13 - gamma*t3*t31 + gamma*t44 + 2*t1*t26*t36*t45 - t10*t16*t39 - t12*t18*t40 + t12 - t14*t6 - t15*t42*t5 + t17*t33*t6 + t19*t21*t7 - 3*t20*t41 + t22*t37*t38 - 3*t23*t29 + t23*t9 + t25*t28 - t28*t40*t45 - t3*t33 + t31*t4 - t32*t43 - t36*t37*t7 + t42 + t14*t17/q**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 235:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = t6**(-2.5)
        t8 = ABx*beta
        t9 = CDx*gamma
        t10 = p**(-2.0)
        t11 = q**(-2.0)
        t12 = t6**(-1.5)
        t13 = F1*t10*t11*t12
        t14 = t13*t8*t9
        t15 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t16 = t15**2
        t17 = t6**(-3.5)
        t18 = F3*t17
        t19 = t16*t18
        t20 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t21 = t20**2
        t22 = F2*t7
        t23 = t22*t3
        t24 = t0*t15*t23
        t25 = ABy*t20
        t26 = alpha*t25
        t27 = 2*t14
        t28 = CDy*t20
        t29 = delta*t28
        t30 = 2*t19
        t31 = 2*t15
        t32 = t23*t25
        t33 = t0*t28
        return np.pi**2.5*(2*ABx*ABy*CDx*CDy*F0*alpha*beta*delta*gamma*t6**(-0.5)/(p**3*q**3) + 2*ABx*ABy*CDy*F1*alpha*beta*delta*t10*t11*t12*t15 + 2*ABx*CDx*F2*beta*gamma*t0*t21*t3*t7 + 2*ABx*F3*beta*t15*t17*t21 - ABx*t2*t31*t32 + 2*ABy*CDx*CDy*F1*alpha*delta*gamma*t10*t11*t12*t15 + 2*ABy*CDy*F2*alpha*delta*t0*t16*t3*t7 - ABy*CDy*alpha*delta*t13 + ABy*F2*alpha*t0*t20*t3*t7 + 2*CDx*F3*gamma*t15*t17*t21 - CDx*t22*t31*t33*t5 + CDy*F2*delta*t0*t20*t3*t7 + (1/2)*F2*t0*t3*t7 + 2*F4*p*q*t16*t21*t6**(-4.5) - t1*t31*t32*t9 - t14 - t18*t21 - t19 - t22*t31*t33*t4*t8 - t24*t8 - t24*t9 - t26*t27 - t26*t30 - t27*t29 - t29*t30)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 236:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = ABy*alpha
        t5 = p + q
        t6 = F1*t5**(-1.5)/p**2
        t7 = t2*t6
        t8 = CDx*delta
        t9 = ABx*ABy*t1
        t10 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t11 = F2*t0*t5**(-2.5)
        t12 = t10*t11
        t13 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t14 = 2*t13**2
        t15 = 2*t13
        t16 = ABx*beta
        t17 = CDx*t3*t6
        t18 = t12*t15
        return np.pi**2.5*(2*F0*t5**(-0.5)*t8*t9/(p**3*q**2) + F3*q*t10*t14*t5**(-3.5) - 2*t10*t16*t17 - t11*t14*t4 - t12 + t15*t17*t4 - t15*t7*t9 + t16*t18 - t18*t8 + t4*t7)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 237:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/(p**2*q**2)
        t8 = ABy*CDz
        t9 = alpha*gamma*t8
        t10 = t7*t9
        t11 = CDx*delta
        t12 = ABx*beta
        t13 = 2*t11*t12
        t14 = F2*t6**(-2.5)
        t15 = t1*t14
        t16 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t17 = ABy*t16
        t18 = t17*t3
        t19 = t15*t18
        t20 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t21 = CDz*t20
        t22 = t0*t14
        t23 = t21*t22
        t24 = t23*t4
        t25 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t26 = 2*t25**2
        t27 = alpha*t17
        t28 = t13*t7
        t29 = 2*t25
        t30 = t12*t29
        t31 = gamma*t21
        t32 = t11*t29
        t33 = t16*t20
        t34 = F3*t6**(-3.5)
        t35 = t33*t34
        t36 = t26*t34
        t37 = t29*t35
        return np.pi**2.5*(-ABx*t14*t18*t2*t29 - CDx*t23*t29*t5 + F0*t13*t6**(-0.5)*t9/(p**3*q**3) + F4*p*q*t26*t33*t6**(-4.5) - t10*t30 + t10*t32 + t10 - t11*t37 + t12*t37 - t13*t22*t3*t33 - t15*t26*t4*t8 + t19*t32 + t19 + t24*t30 - t24 + t27*t28 - t27*t36 - t28*t31 + t31*t36 - t35)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 238:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F2*t6**(-2.5)
        t8 = t3*t7
        t9 = t0*t8
        t10 = ABx*beta
        t11 = CDx*t10
        t12 = delta*t11
        t13 = F1*t6**(-1.5)/(p**2*q**2)
        t14 = t12*t13
        t15 = CDy*gamma
        t16 = ABy*t15
        t17 = alpha*t16
        t18 = t13*t17
        t19 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t20 = t19**2
        t21 = F3*t6**(-3.5)
        t22 = t20*t21
        t23 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t24 = t23**2
        t25 = t21*t24
        t26 = ABy*t23
        t27 = t1*t8
        t28 = CDx*t19
        t29 = t0*t4*t7
        t30 = t10*t19
        t31 = t30*t9
        t32 = t15*t23
        t33 = 2*t24
        t34 = alpha*t26
        t35 = 2*t18
        t36 = 2*t32
        t37 = delta*t28
        t38 = 2*t25
        t39 = 2*t22
        t40 = 2*t26
        t41 = t28*t7
        return np.pi**2.5*(-ABx*t19*t2*t40*t8 - 2*CDy*t0*t23*t41*t5 + 2*F0*t12*t17*t6**(-0.5)/(p**3*q**3) + F4*p*q*t20*t33*t6**(-4.5) + t1*t4*t40*t41 - t11*t29*t33 + 2*t14*t34 - t14*t36 + t14 - 2*t16*t20*t27 + t18 - t22 - t25 + t26*t27 + t28*t29 - t30*t35 + t30*t38 + t31*t36 - t31 + t32*t39 - t32*t9 - t34*t39 + t35*t37 - t37*t38 + (1/2)*t9)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 239:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = ABy*alpha
        t7 = q**(-2.0)
        t8 = p + q
        t9 = F1*t8**(-1.5)/p**2
        t10 = t7*t9
        t11 = t10*t6
        t12 = ABx*beta
        t13 = CDx*t11
        t14 = F0*t12*t6*t8**(-0.5)/p**3
        t15 = 2*CDx**2
        t16 = delta*t15
        t17 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t18 = t17**3
        t19 = F3*t8**(-3.5)
        t20 = 2*t19
        t21 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t22 = ABx*t21
        t23 = beta*t22
        t24 = t3*t9
        t25 = F2*t8**(-2.5)
        t26 = t0*t25
        t27 = CDx*t21
        t28 = t26*t27
        t29 = t25*t3
        t30 = t1*t29
        t31 = t17*t6
        t32 = gamma*t3
        t33 = t17**2
        t34 = 2*t33
        t35 = CDx*t26*t34*t6
        t36 = 2*CDx
        t37 = t12*t31*t36
        t38 = gamma*t10
        t39 = t16*t38
        t40 = t17*t21
        t41 = t26*t40
        t42 = t20*t33
        t43 = t27*t42
        t44 = ABx*t36*t40
        return np.pi**2.5*(-ABx*ABy*t2*t29*t34 + 2*F4*p*q*t18*t21*t8**(-4.5) + delta*t10*t37 - delta*t13 - delta*t43 + gamma*t13 + gamma*t30*t44 + gamma*t43 + gamma*t14*t16/q**3 - t1*t25*t4*t44 + t11*t12 - t14*t7 - t15*t41*t5 - t18*t20*t6 - 3*t19*t40 - t22*t30 + t23*t24 - t23*t39 + t23*t42 - t24*t31 + 3*t26*t3*t31 - t28*t32 + t28*t4 + t31*t39 - t32*t35 + t35*t4 - t37*t38 + t41)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 240:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = p + q
        t4 = t3**(-1.5)
        t5 = F1*t0*t4
        t6 = t3**(-0.5)
        t7 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        return np.pi**2.5*t0*(-2*ABx**2*F0*t1*t2*t6/p**2 + 2*ABx*F1*beta*t0*t4*t7 - 2*ABx*alpha*t5*t7 + F0*t0*t2*t6 + 2*F2*q*t3**(-2.5)*t7**2 - t5)*np.exp(-delta*gamma*rCD*t2 - rAB*t0*t1)
    if case_id == 241:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = p**(-2.0)
        t8 = CDz*t3*t6*t7
        t9 = q**(-2.0)
        t10 = t4**(-0.5)
        t11 = 2*ABx**2*t1
        t12 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t13 = t4**(-2.5)
        t14 = F2*t0*t12*t13
        t15 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t16 = t15**2
        t17 = 2*ABx*t15
        return np.pi**2.5*(2*ABx*CDz*F1*beta*gamma*t15*t2*t5*t7 + 2*ABx*F2*beta*t0*t12*t13*t15 + CDz*F0*gamma*t10*t7*t9 - CDz*F0*gamma*t10*t11*t9/p**3 + 2*CDz*F2*gamma*t0*t13*t16 + F1*t0*t12*t2*t5 + 2*F3*q*t12*t16*t4**(-3.5) - alpha*t14*t17 - alpha*t17*t8 - t11*t12*t2*t6*t7 - t14 - t8)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 242:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = p**(-2.0)
        t8 = CDy*t3*t6*t7
        t9 = q**(-2.0)
        t10 = t4**(-0.5)
        t11 = 2*ABx**2*t1
        t12 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t13 = t4**(-2.5)
        t14 = F2*t0*t12*t13
        t15 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t16 = t15**2
        t17 = 2*ABx*t15
        return np.pi**2.5*(2*ABx*CDy*F1*beta*gamma*t15*t2*t5*t7 + 2*ABx*F2*beta*t0*t12*t13*t15 + CDy*F0*gamma*t10*t7*t9 - CDy*F0*gamma*t10*t11*t9/p**3 + 2*CDy*F2*gamma*t0*t13*t16 + F1*t0*t12*t2*t5 + 2*F3*q*t12*t16*t4**(-3.5) - alpha*t14*t17 - alpha*t17*t8 - t11*t12*t2*t6*t7 - t14 - t8)*np.exp(-delta*rCD*t3 - rAB*t0*t1)
    if case_id == 243:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = gamma*t2
        t4 = p + q
        t5 = t4**(-1.5)
        t6 = F1*t5
        t7 = t2*t6
        t8 = p**(-2.0)
        t9 = ABx*t8
        t10 = t3*t6
        t11 = q**(-2.0)
        t12 = t4**(-0.5)
        t13 = ABx**2*beta
        t14 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t15 = t4**(-2.5)
        t16 = F2*t15
        t17 = t14**2
        t18 = 2*alpha*t14
        return np.pi**2.5*(2*ABx*CDx*F1*beta*gamma*t14*t2*t5*t8 + ABx*F1*alpha*t2*t5*t8 + 2*ABx*F2*beta*t0*t15*t17 - 2*ABx*t1*t16*t17 - 2*CDx*F0*alpha*gamma*t11*t12*t13/p**3 + CDx*F0*gamma*t11*t12*t8 + 2*CDx*F2*gamma*t0*t15*t17 - CDx*t10*t18*t9 - CDx*t10*t8 + F1*t0*t14*t2*t5 + 2*F3*q*t14**3*t4**(-3.5) - beta*t7*t9 - 3*t0*t14*t16 - t13*t18*t7*t8)*np.exp(-beta*rAB*t1 - delta*rCD*t3)
    if case_id == 244:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = p**(-2.0)
        t7 = CDz*t6
        t8 = t3*t5*t7
        t9 = F0*delta*t4**(-0.5)/q**2
        t10 = 2*ABx**2*t1
        t11 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t12 = t0*t11
        t13 = F2*t4**(-2.5)
        t14 = t12*t13
        t15 = t2*t5
        t16 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t17 = 2*t16**2
        t18 = 2*ABx*t16
        t19 = t18*t8
        t20 = t14*t18
        return np.pi**2.5*(-CDz*delta*t0*t13*t17 + CDz*t10*t9/p**3 + F3*q*t11*t17*t4**(-3.5) + alpha*t19 - alpha*t20 - beta*t19 + beta*t20 - t10*t11*t15*t6 + t12*t15 - t14 - t7*t9 + t8)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 245:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = gamma*t3
        t5 = q**(-2.0)
        t6 = (1/2)*t5
        t7 = p**(-2.0)
        t8 = p + q
        t9 = F0*t8**(-0.5)
        t10 = t7*t9
        t11 = (1/2)*t2
        t12 = F1*t8**(-1.5)
        t13 = t12*t7
        t14 = t0*t12
        t15 = F2*t8**(-2.5)
        t16 = t0*t15
        t17 = ABx**2*t1
        t18 = t17*t5
        t19 = t13*t18
        t20 = CDz**2
        t21 = delta*t20
        t22 = gamma*t5
        t23 = t13*t22
        t24 = t9/p**3
        t25 = gamma/q**3
        t26 = 2*delta
        t27 = t20*t26
        t28 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t29 = t28**2
        t30 = F3*t8**(-3.5)
        t31 = t29*t30
        t32 = t0*(Az*alpha + Bz*beta) - t2*(Cz*gamma + Dz*delta)
        t33 = t32**2
        t34 = t30*t33
        t35 = t16*t29
        t36 = t2*t33
        t37 = ABx*t28
        t38 = beta*t37
        t39 = t13*t2
        t40 = alpha*t37
        t41 = t16*t2
        t42 = t40*t41
        t43 = CDz*t32
        t44 = t14*t43
        t45 = t16*t3*t43
        t46 = t38*t41
        t47 = t23*t27
        t48 = 2*gamma
        t49 = t43*t48
        t50 = 2*t34
        t51 = t31*t43
        t52 = 2*t45
        return np.pi**2.5*(2*F4*p*q*t29*t33*t8**(-4.5) - delta*t44*t5 - gamma*t41*t43 - t10*t21*t25 + t10*t6 - t11*t13 + t11*t16 - t14*t6 + t15*t36 - 2*t16*t17*t36 + t17*t24*t25*t27 - t18*t24 + t19*t26*t43 - t19*t49 + t19 - 2*t20*t35*t4 + t21*t23 + t22*t44 - t26*t51 - t31 - t34 + t35 + t38*t39 - t38*t47 + t38*t50 - t38*t52 - t39*t40 + t40*t47 - t40*t50 + t40*t52 - t42*t49 + t42 + t45 + t46*t49 - t46 + t48*t51)*np.exp(-rAB*t0*t1 - rCD*t4)
    if case_id == 246:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p**(-2.0)
        t7 = p + q
        t8 = F1*t7**(-1.5)/q**2
        t9 = t6*t8
        t10 = CDy*gamma
        t11 = CDz*delta*t10
        t12 = t11*t9
        t13 = F0*t11*t7**(-0.5)/q**3
        t14 = ABx**2
        t15 = 2*alpha
        t16 = beta*t14*t15
        t17 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t18 = t10*t17
        t19 = t0*t18
        t20 = F2*t7**(-2.5)
        t21 = t0*t20
        t22 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t23 = CDz*t22
        t24 = t23*t4
        t25 = t21*t24
        t26 = t20*t3
        t27 = t19*t26
        t28 = delta*t23
        t29 = t28*t8
        t30 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t31 = 2*t30**2
        t32 = ABx*t30
        t33 = 2*beta
        t34 = t32*t33
        t35 = t17*t22
        t36 = F3*t7**(-3.5)
        t37 = t35*t36
        t38 = t26*t35
        t39 = t31*t36
        t40 = 2*t1*t32
        t41 = t32*t37
        return np.pi**2.5*(-CDy*CDz*t21*t31*t5 + F4*p*q*t31*t35*t7**(-4.5) - t0*t29 + t12*t15*t32 - t12*t34 + t12 - t13*t6 - 2*t14*t2*t38 - t15*t41 - t16*t18*t9 + t16*t29*t6 - t18*t26*t40 + t18*t39 + t19*t8 + t20*t24*t40 - t25*t34 + t25 + t27*t34 - t27 - t28*t39 + t33*t41 - t37 + t38 + t13*t16/p**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 247:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/q**2
        t8 = p**(-2.0)
        t9 = CDz*delta
        t10 = t8*t9
        t11 = t10*t7
        t12 = ABx*t11
        t13 = CDx*gamma
        t14 = F0*t13*t6**(-0.5)/q**3
        t15 = 2*ABx**2
        t16 = alpha*t15
        t17 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t18 = t17**3
        t19 = F3*t6**(-3.5)
        t20 = 2*t19
        t21 = F2*t6**(-2.5)
        t22 = t21*t3
        t23 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t24 = ABx*t23
        t25 = t22*t24
        t26 = CDx*t23
        t27 = gamma*t26
        t28 = t0*t7
        t29 = beta*t0
        t30 = t0*t21
        t31 = t17*t9
        t32 = t17**2
        t33 = 2*t32
        t34 = t22*t33*t9
        t35 = ABx*t1
        t36 = ABx*t29
        t37 = 2*ABx*t13*t31
        t38 = beta*t7*t8
        t39 = t16*t38
        t40 = t17*t23
        t41 = t22*t40
        t42 = t20*t32
        t43 = t24*t42
        t44 = 2*CDx*t21*t4*t40
        return np.pi**2.5*(-CDx*CDz*t30*t33*t5 + 2*F4*p*q*t18*t23*t6**(-4.5) - alpha*t12 + alpha*t37*t7*t8 - alpha*t43 + beta*t12 + beta*t43 + beta*t14*t16*t9/p**3 + 3*t0*t22*t31 + t1*t25 - t10*t14 + t11*t13 - t15*t2*t41 - t18*t20*t9 - 3*t19*t40 - t25*t29 - t26*t30*t4 + t27*t28 - t27*t39 + t27*t42 - t28*t31 + t31*t39 + t34*t35 - t34*t36 - t35*t44 + t36*t44 - t37*t38 + t41)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 248:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = p**(-2.0)
        t7 = CDy*t6
        t8 = t3*t5*t7
        t9 = F0*delta*t4**(-0.5)/q**2
        t10 = 2*ABx**2*t1
        t11 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t12 = t0*t11
        t13 = F2*t4**(-2.5)
        t14 = t12*t13
        t15 = t2*t5
        t16 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t17 = 2*t16**2
        t18 = 2*ABx*t16
        t19 = t18*t8
        t20 = t14*t18
        return np.pi**2.5*(-CDy*delta*t0*t13*t17 + CDy*t10*t9/p**3 + F3*q*t11*t17*t4**(-3.5) + alpha*t19 - alpha*t20 - beta*t19 + beta*t20 - t10*t11*t15*t6 + t12*t15 - t14 - t7*t9 + t8)*np.exp(-gamma*rCD*t3 - rAB*t0*t1)
    if case_id == 249:
        t0 = p**(-1.0)
        t1 = beta*t0
        t2 = alpha*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p**(-2.0)
        t7 = p + q
        t8 = F1*t7**(-1.5)/q**2
        t9 = t6*t8
        t10 = CDz*gamma
        t11 = CDy*delta*t10
        t12 = t11*t9
        t13 = F0*t11*t7**(-0.5)/q**3
        t14 = ABx**2
        t15 = 2*alpha
        t16 = beta*t14*t15
        t17 = F2*t7**(-2.5)
        t18 = t0*t17
        t19 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t20 = CDy*t19
        t21 = t20*t4
        t22 = t18*t21
        t23 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t24 = t10*t23
        t25 = t0*t24
        t26 = delta*t20
        t27 = t26*t8
        t28 = t17*t3
        t29 = t25*t28
        t30 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t31 = 2*t30**2
        t32 = ABx*t30
        t33 = t15*t32
        t34 = 2*beta
        t35 = t19*t23
        t36 = F3*t7**(-3.5)
        t37 = t35*t36
        t38 = t28*t35
        t39 = t31*t36
        t40 = 2*t1*t32
        t41 = t32*t37
        return np.pi**2.5*(-CDy*CDz*t18*t31*t5 + F4*p*q*t31*t35*t7**(-4.5) - t0*t27 - t12*t32*t34 + t12*t33 + t12 - t13*t6 - 2*t14*t2*t38 - t15*t41 - t16*t24*t9 + t16*t27*t6 - t17*t21*t40 + t22*t33 + t22 + t24*t28*t40 + t24*t39 + t25*t8 - t26*t39 - t29*t33 - t29 + t34*t41 - t37 + t38 + t13*t16/p**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 250:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = gamma*t3
        t5 = q**(-2.0)
        t6 = (1/2)*t5
        t7 = p**(-2.0)
        t8 = p + q
        t9 = F0*t8**(-0.5)
        t10 = t7*t9
        t11 = (1/2)*t2
        t12 = F1*t8**(-1.5)
        t13 = t12*t7
        t14 = t0*t12
        t15 = F2*t8**(-2.5)
        t16 = t0*t15
        t17 = ABx**2*t1
        t18 = t17*t5
        t19 = t13*t18
        t20 = CDy**2
        t21 = delta*t20
        t22 = gamma*t5
        t23 = t13*t22
        t24 = t9/p**3
        t25 = gamma/q**3
        t26 = 2*delta
        t27 = t20*t26
        t28 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t29 = t28**2
        t30 = F3*t8**(-3.5)
        t31 = t29*t30
        t32 = t0*(Ay*alpha + By*beta) - t2*(Cy*gamma + Dy*delta)
        t33 = t32**2
        t34 = t30*t33
        t35 = t16*t29
        t36 = t2*t33
        t37 = ABx*t28
        t38 = beta*t37
        t39 = t13*t2
        t40 = alpha*t37
        t41 = t16*t2
        t42 = t40*t41
        t43 = CDy*t32
        t44 = t14*t43
        t45 = t16*t3*t43
        t46 = t38*t41
        t47 = t23*t27
        t48 = 2*gamma
        t49 = t43*t48
        t50 = 2*t34
        t51 = t31*t43
        t52 = 2*t45
        return np.pi**2.5*(2*F4*p*q*t29*t33*t8**(-4.5) - delta*t44*t5 - gamma*t41*t43 - t10*t21*t25 + t10*t6 - t11*t13 + t11*t16 - t14*t6 + t15*t36 - 2*t16*t17*t36 + t17*t24*t25*t27 - t18*t24 + t19*t26*t43 - t19*t49 + t19 - 2*t20*t35*t4 + t21*t23 + t22*t44 - t26*t51 - t31 - t34 + t35 + t38*t39 - t38*t47 + t38*t50 - t38*t52 - t39*t40 + t40*t47 - t40*t50 + t40*t52 - t42*t49 + t42 + t45 + t46*t49 - t46 + t48*t51)*np.exp(-rAB*t0*t1 - rCD*t4)
    if case_id == 251:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = gamma*t3
        t5 = delta*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/q**2
        t8 = p**(-2.0)
        t9 = CDy*delta
        t10 = t8*t9
        t11 = t10*t7
        t12 = ABx*t11
        t13 = CDx*gamma
        t14 = F0*t13*t6**(-0.5)/q**3
        t15 = 2*ABx**2
        t16 = alpha*t15
        t17 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t18 = t17**3
        t19 = F3*t6**(-3.5)
        t20 = 2*t19
        t21 = F2*t6**(-2.5)
        t22 = t21*t3
        t23 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t24 = ABx*t23
        t25 = t22*t24
        t26 = CDx*t23
        t27 = gamma*t26
        t28 = t0*t7
        t29 = beta*t0
        t30 = t0*t21
        t31 = t17*t9
        t32 = t17**2
        t33 = 2*t32
        t34 = t22*t33*t9
        t35 = ABx*t1
        t36 = ABx*t29
        t37 = 2*ABx*t13*t31
        t38 = beta*t7*t8
        t39 = t16*t38
        t40 = t17*t23
        t41 = t22*t40
        t42 = t20*t32
        t43 = t24*t42
        t44 = 2*CDx*t21*t4*t40
        return np.pi**2.5*(-CDx*CDy*t30*t33*t5 + 2*F4*p*q*t18*t23*t6**(-4.5) - alpha*t12 + alpha*t37*t7*t8 - alpha*t43 + beta*t12 + beta*t43 + beta*t14*t16*t9/p**3 + 3*t0*t22*t31 + t1*t25 - t10*t14 + t11*t13 - t15*t2*t41 - t18*t20*t9 - 3*t19*t40 - t25*t29 - t26*t30*t4 + t27*t28 - t27*t39 + t27*t42 - t28*t31 + t31*t39 + t34*t35 - t34*t36 - t35*t44 + t36*t44 - t37*t38 + t41)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 252:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = q**(-1.0)
        t3 = delta*t2
        t4 = p + q
        t5 = F1*t4**(-1.5)
        t6 = t2*t5
        t7 = p**(-2.0)
        t8 = ABx*t7
        t9 = t6*t8
        t10 = CDx*t7
        t11 = t3*t5
        t12 = F0*delta*t4**(-0.5)/q**2
        t13 = ABx**2*alpha
        t14 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t15 = t0*t14
        t16 = F2*t4**(-2.5)
        t17 = 2*t14**2*t16
        t18 = ABx*t17
        t19 = CDx*t11*t8
        t20 = 2*beta*t14
        return np.pi**2.5*(2*CDx*beta*t12*t13/p**3 - CDx*delta*t0*t17 + 2*F3*q*t14**3*t4**(-3.5) + 2*alpha*t14*t19 + alpha*t9 + beta*t0*t18 - beta*t9 - t1*t18 + t10*t11 - t10*t12 - t13*t20*t6*t7 - 3*t15*t16 + t15*t6 - t19*t20)*np.exp(-beta*rAB*t1 - gamma*rCD*t3)
    if case_id == 253:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/q**2
        t8 = p**(-2.0)
        t9 = CDz*gamma
        t10 = t8*t9
        t11 = t10*t7
        t12 = ABx*t11
        t13 = alpha*t12
        t14 = CDx*delta
        t15 = beta*t12
        t16 = F0*t14*t6**(-0.5)/q**3
        t17 = 2*ABx**2
        t18 = alpha*beta*t17
        t19 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t20 = t19**3
        t21 = F3*t6**(-3.5)
        t22 = 2*t21
        t23 = F2*t6**(-2.5)
        t24 = t23*t3
        t25 = t0*(Az*alpha + Bz*beta) - t3*(Cz*gamma + Dz*delta)
        t26 = ABx*t25
        t27 = t24*t26
        t28 = CDx*t25
        t29 = t0*t23
        t30 = t29*t4
        t31 = t0*t9
        t32 = t19*t31
        t33 = delta*t28
        t34 = t33*t7
        t35 = ABx*t1
        t36 = t19**2
        t37 = 2*t36
        t38 = t24*t37
        t39 = ABx*beta
        t40 = 2*t14*t19
        t41 = t19*t25
        t42 = t24*t41
        t43 = t22*t36
        t44 = t26*t43
        t45 = 2*CDx*t41
        return np.pi**2.5*(-CDx*CDz*t29*t37*t5 + 2*F4*p*q*t20*t25*t6**(-4.5) - alpha*t44 - beta*t0*t27 + beta*t44 - t0*t34 + t1*t27 - t10*t16 + t11*t14 - t11*t18*t19 + t13*t40 + t13 - t15*t40 - t15 - t17*t2*t42 + t18*t34*t8 + t20*t22*t9 - 3*t21*t41 + t23*t35*t4*t45 - 3*t24*t32 + t28*t30 - t30*t39*t45 + t31*t38*t39 + t32*t7 - t33*t43 - t35*t38*t9 + t42 + t16*t18*t9/p**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 254:
        t0 = p**(-1.0)
        t1 = alpha*t0
        t2 = beta*t1
        t3 = q**(-1.0)
        t4 = delta*t3
        t5 = gamma*t4
        t6 = p + q
        t7 = F1*t6**(-1.5)/q**2
        t8 = p**(-2.0)
        t9 = CDy*gamma
        t10 = t8*t9
        t11 = t10*t7
        t12 = ABx*t11
        t13 = alpha*t12
        t14 = CDx*delta
        t15 = beta*t12
        t16 = F0*t14*t6**(-0.5)/q**3
        t17 = 2*ABx**2
        t18 = alpha*beta*t17
        t19 = t0*(Ax*alpha + Bx*beta) - t3*(Cx*gamma + Dx*delta)
        t20 = t19**3
        t21 = F3*t6**(-3.5)
        t22 = 2*t21
        t23 = F2*t6**(-2.5)
        t24 = t23*t3
        t25 = t0*(Ay*alpha + By*beta) - t3*(Cy*gamma + Dy*delta)
        t26 = ABx*t25
        t27 = t24*t26
        t28 = CDx*t25
        t29 = t0*t23
        t30 = t29*t4
        t31 = t0*t9
        t32 = t19*t31
        t33 = delta*t28
        t34 = t33*t7
        t35 = ABx*t1
        t36 = t19**2
        t37 = 2*t36
        t38 = t24*t37
        t39 = ABx*beta
        t40 = 2*t14*t19
        t41 = t19*t25
        t42 = t24*t41
        t43 = t22*t36
        t44 = t26*t43
        t45 = 2*CDx*t41
        return np.pi**2.5*(-CDx*CDy*t29*t37*t5 + 2*F4*p*q*t20*t25*t6**(-4.5) - alpha*t44 - beta*t0*t27 + beta*t44 - t0*t34 + t1*t27 - t10*t16 + t11*t14 - t11*t18*t19 + t13*t40 + t13 - t15*t40 - t15 - t17*t2*t42 + t18*t34*t8 + t20*t22*t9 - 3*t21*t41 + t23*t35*t4*t45 - 3*t24*t32 + t28*t30 - t30*t39*t45 + t31*t38*t39 + t32*t7 - t33*t43 - t35*t38*t9 + t42 + t16*t18*t9/p**3)*np.exp(-rAB*t2 - rCD*t5)
    if case_id == 255:
        t0 = p**(-1.0)
        t1 = alpha*beta
        t2 = q**(-1.0)
        t3 = delta*gamma
        t4 = t2*t3
        t5 = q**(-2.0)
        t6 = (1/2)*t5
        t7 = p**(-2.0)
        t8 = p + q
        t9 = F0*t8**(-0.5)
        t10 = t7*t9
        t11 = F1*t8**(-1.5)
        t12 = t11*t7
        t13 = t12*t2
        t14 = t0*t11
        t15 = t8**(-2.5)
        t16 = F2*t0*t15*t2
        t17 = ABx*alpha
        t18 = t12*t5
        t19 = CDx*t18
        t20 = t17*t19
        t21 = ABx*beta
        t22 = t19*t21
        t23 = ABx**2*t1
        t24 = t18*t23
        t25 = CDx**2
        t26 = t25*t3
        t27 = t18*t26
        t28 = t23*t9/p**3
        t29 = t26/q**3
        t30 = t0*(Ax*alpha + Bx*beta) - t2*(Cx*gamma + Dx*delta)
        t31 = t30**2
        t32 = F3*t8**(-3.5)
        t33 = F2*t15*t31
        t34 = t0*t33
        t35 = 2*t30**3*t32
        t36 = CDx*t35
        t37 = t13*t30
        t38 = CDx*t30
        t39 = gamma*t38
        t40 = t14*t5
        t41 = 3*t16
        t42 = t17*t30
        t43 = t21*t30
        t44 = delta*t38
        t45 = 2*t34
        t46 = t2*t45
        t47 = CDx*t46
        t48 = t17*t47
        t49 = t21*t47
        t50 = 2*t27
        t51 = 2*t24
        return np.pi**2.5*(2*F4*p*q*t30**4*t8**(-4.5) - delta*t20 + delta*t22 - delta*t36 + delta*t48 - delta*t49 + gamma*t20 - gamma*t22 + gamma*t36 - gamma*t48 + gamma*t49 - t10*t29 + t10*t6 - 1/2*t13 - t14*t6 + (3/2)*t16 - t17*t35 - t17*t37 + t2*t33 + t21*t35 + t21*t37 - t23*t46 + t24 - t25*t4*t45 + t27 + 2*t28*t29 - t28*t5 - 6*t31*t32 + t34 + t39*t40 - t39*t41 - t39*t51 - t40*t44 + t41*t42 - t41*t43 + t41*t44 + t42*t50 - t43*t50 + t44*t51)*np.exp(-rAB*t0*t1 - rCD*t4)
    raise KeyError(case_id)
