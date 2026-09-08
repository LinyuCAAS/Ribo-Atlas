#!/bin/bash
#DSUB -R "cpu=8;mem=5GB"
#DSUB -e err.txt


samp=Adrenal_gland_1

cutadapt  -q 20 --match-read-wildcards --max-n 0.25 -a TGGAATTCTCGG -o  $samp.cut_adaptor.fq.gz --minimum-length=24 --maximum-length=35  $samp.fq1.gz


gzip -dc $samp.cut_adaptor.fq.gz >  $samp.cut_adaptor.fq


bowtie -x  Database/rRNA_SILVA/Merged.rRNA.U2T.fasta   --al  $samp.mapped_rRNA.fq  --un  $samp.unmapped_rRNA.fq  -p 8  $samp.cut_adaptor.fq


bowtie -x  Database/GtRNAdb/Merged.tRNA.U2T.fasta   --al  $samp.mapped_tRNA.fq  --un  $samp.unmapped_tRNA.fq  -p 8  $samp.unmapped_rRNA.fq


STAR \
   --runThreadN 8 \
   --outFilterType BySJout \
   --outFilterMismatchNmax 2 \
   --outFilterMultimapNmax 1 \
   --genomeDir Database/Sus_scrofa_V113/STAR_for_Ribo \
   --readFilesIn $samp.unmapped_tRNA.fq \
   --outFileNamePrefix $samp \
   --outSAMtype BAM SortedByCoordinate \
   --quantMode TranscriptomeSAM GeneCounts \
   --outSAMattributes All \
   --outSAMattrRGline ID:1 LB:Ribo_seq PL:ILLUMINA SM:$samp \
   --outBAMcompression 6 \
   --outReadsUnmapped Fastx





