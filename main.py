#SCRIPT WHICH CLEAN SEQUENCES BY REPEAT AND AA AND SIZE
#modules
import sys
import argparse

from pathlib import Path

def parse_arguments(): #Functions to generate program options
    #Description of script function
    desc = "Bioinformatic tool to clean, get quality of annotated genome, annotate with DIAMOND and FANTASIA pipeline"
    #Store arguments in variable parser
    parser = argparse.ArgumentParser(description=desc)

    #Required Arguments
    help_input = '''(Required) Genome fasta, gff file and lineage (Optional)'''

    parser.add_argument("--input", "-i", type=str, nargs=2, 
                        help=help_input,
                        required=True)
    
    help_output_dir = '''(Required) Output dir'''

    parser.add_argument("--output", "-o", type=str,nargs='+',
                        help=help_output_dir,
                        required=True)
    
    #Optional Arguments
    help_statistics_agat = '''(Optional) Flag On/Off if argument present'''

    parser.add_argument("--agat", "-a", action="store_true",
                        help = help_statistics_agat)
    
    help_longest_protein_isoform = '''(Optional) GFF with longest protein isoform'''

    parser.add_argument("--lisof", "-lf", action="store_true",
                        help=help_longest_protein_isoform)

    help_clean_sequence = '''(Optional) Filter sequences by valid amino acids, repeated sequences, size. 
                                Size Required'''

    parser.add_argument("--clean", "-c", action="store_true",
                        help=help_clean_sequence)

    help_size = '''(Required if --clean is enabled) Max sequences size accepted in fasta'''

    parser.add_argument("--size", "-s", type=int, 
                        help=help_size, default=None)

    help_run_busco = '''(Optional) Run Busco analysis'''

    parser.add_argument("--rbusco", "-rb", action="store_true",
                        help=help_run_busco)

    help_lineage ='''(Required if --rbusco enable) Species lineage for busco analysis'''

    parser.add_argument("--lineage", "-l", type=str,
                        help=help_lineage, default=None, const="embryophyte")

    help_run_diamond = '''(Optional) Run diamond functional annotation'''

    parser.add_argument("--rdiamond", "-rd", action="store_true",
                        help=help_run_diamond)

    help_yml_generator = '''(Required if --rdiamond enable) Create yml file for ahrd command'''

    parser.add_argument("--yml", "-y", action="store_true",
                        help=help_yml_generator)
    
    help_run_FANTASIA = '''(Optional) Run FANTASIA annotation pipeline !!One at time install
                            Recommended longest isoform protein'''
    parser.add_argument("--rfanta", "-rf", action="store_true",
                        help=help_run_FANTASIA)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit("Arguments not found")

    return parser.parse_args()



def get_arguments():
    parser = parse_arguments()

    #variables
    input = parser.input
    output = Path(parser.output)
    statistics_agat = parser.agat
    longest_protein_isoform = parser.lisof
    clean_sequence = parser.clean
    size = parser.size
    run_busco = parser.rbusco
    lineage = parser.lineage
    run_diamong = parser.rdiamond
    yml_generator = parser.yml
    run_fantasia = parser.rfanta
    
    #conditionals
    if not output.exists():
        output.mkdir(parents=True)
    
    if clean_sequence:
        if size == None:
            sys.exit("Size required when clean sequence enable")
        elif (size > 0) and (size != None):
            print("Size argument correct")
        else:
            sys.exit("Size is not correct")
    

    if run_busco:
        if (lineage is None) or not (yml_generator):
            sys.exit("Lineage and yml required when run busco enable")
        elif lineage == "embryophyta":
            print("lineage set to default")
        else:
            print(f"lineage set to {lineage}")
    return {
        "input": input,
        "output": output,
        "agat": statistics_agat,
        "lisof": longest_protein_isoform,
        "clean": clean_sequence,
        "size": size,
        "rbusco": run_busco,
        "lineage": lineage,
        "rdiamond": run_diamong,
        "yml": yml_generator,
        "rfanta": run_fantasia
        }
    
#Function to get unique sequences, and correct unvalid aa. It accepts fasta file and uses yield instead of return 
#yield optimizes resources
def get_unique_seq_treated(fasta_in):
    #variables to control statements and pass to next function
    header = None
    names = set()
    unique = False
    seq_lines = []
    aa_set = set('ARNDCEQGHILKMFPSTWYVX') #create set with all valid amino acids 
    #open input file
    with open(fasta_in, 'r') as fin:
        
        #iterates input 
        for lines in fin:
            #process lines
            lines = lines.strip()
            #get the first line of each sequence
            if lines.startswith('>'):
                #check if previous line has a header and if unique
                if header is not None and unique:
                    yield (header, '\n'.join(seq_lines))
                #create header
                header = lines.split('>')[1]
                #if header not in set names: unique and add it
                if header not in names:
                    unique = True
                    names.add(header)
                else:
                    unique = False
                seq_lines = []
            else:
                if unique: #when unique flag is true
                    lines = lines.strip()
                    sequence_aa = '' #create empy variable to store new sequence
                    for aa in lines: #iterate old sequence
                        if aa in aa_set: #search for valid aa in aa set
                            sequence_aa += aa #add valid aa
                        else: 
                            aa = 'X' #change invalid aa into valid
                            sequence_aa += aa #add new valid aa
                    seq_lines.append(sequence_aa)
        if header is not None and unique:
            yield (header, '\n'.join(seq_lines))
                
# Function to create a dictionary with sequences names and size 
def get_seq_size(unique_seqs):
    sizes = {} #variable to store header, seqs, size
    for header, sequences in unique_seqs:
        if header not in sizes:
            sizes[header] = len(sequences)
        # else:
        #     sizes[header] = len(sequences) #line due to an error in yield in which I use wrong iteration and get incomplete sequence.
            
    return sizes
        
    
#Function which filters based on size. The function gets the value from previous functions "get unique seq" & "get seq size"
def filter_sequences(size, unique_seqs, max_size, fasta_out):
    with open(fasta_out, 'w') as output:
        for header, sequences in unique_seqs:
            seqs_size = size.get(header, 0)
            if seqs_size < max_size:
                output.write(f'>{header}\n{sequences}\n')
                			
          
# if __name__ == '__main__':
#     fasta_in = sys.argv[1]
#     fasta_out = sys.argv[2]
#     max_size = int(sys.argv[3])
#     unique_iter = get_unique_seq_treated(fasta_in)
#     seq_sizes = get_seq_size(unique_iter)
#     unique_iter = get_unique_seq_treated(fasta_in)
#     filter_seqs = filter_sequences(seq_sizes, unique_iter, max_size, fasta_out)
    
    