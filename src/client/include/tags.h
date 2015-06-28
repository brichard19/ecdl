#ifndef _TAGS_H
#define _TAGS_H

enum MessageTags {
    TAG_START_A = 0,
    TAG_START_B,
    TAG_END_X,
    TAG_END_Y,
    TAG_G_X,
    TAG_G_Y,
    TAG_Q_X,
    TAG_Q_Y,
    TAG_CURVE_P,
    TAG_CURVE_A,
    TAG_CURVE_B,
    TAG_CURVE_N,
    TAG_R_COUNT,
    TAG_R_X,
    TAG_R_Y,
    TAG_DISTINGUISHED_BITS
};

enum {
    MSG_TYPE_REGISTER,
    MSG_TYPE_SUBMIT,
    MSG_TYPE_STATUS
};



#endif
