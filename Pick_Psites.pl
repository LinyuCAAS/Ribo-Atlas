use strict;



my %PST=(
        27 => 11,
        28 => 12,
        29 => 12,
        30 => 12,
        31 => 12,
);


open IN,"samtools view All_merged_Ribo.Psites.bam |" or die;
while(<IN>){
	chomp;
	my @a=split /\t/,$_;
	my $sz=length($a[9]);
	next unless (exists $PST{$sz});
	my $flag=$a[1]&16;
	if($flag==0){
		my $offs=$a[3] + $PST{$sz};
		print "$a[2]\t$a[3]\t$offs\t+\t$PST{$sz}\n";
	}elsif($flag==16){
		my $offs=$a[3] +$sz - 1 - $PST{$sz};
                print "$a[2]\t$a[3]\t$offs\t-\t$PST{$sz}\n";
	}
}
