#include "imageLib.h"
#include <cstdio>
void ReadFilePFM(CFloatImage&, const char*);
int main(int argc,char** argv) {
 if(argc!=3) return 2;
 CFloatImage img; ReadFilePFM(img,argv[1]); CShape s=img.Shape();
 FILE* f=fopen(argv[2],"wb"); if(!f)return 3;
 for(int y=0;y<s.height;y++) fwrite(img.PixelAddress(0,y,0),sizeof(float),s.width,f);
 return fclose(f);
}
