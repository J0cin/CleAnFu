'''DESCRIPTION
- Eliminate repeated sequences
- Change Amino acids which are not part of normal set
- Filter Fasta sequences by a selected length 
'''
import argparse #library to parse arguments
import sys #library to interact with console 

from pathlib import Path

def parse_arguments():
    desc = ("Filter a FASTA file of proteins into a clean FASTA file by "
            "eliminating duplicate sequences, replacing non-standard amino acids "
            "with 'X', and optionally filtering by sequence length.")
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="(Required) Protein FASTA file")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="(Required) Output FASTA file")
    parser.add_argument("--overlength", "-l", action="store_true",
                        help="(Optional) Enable filtering by sequence length")
    parser.add_argument("--size", "-s", type=int, default=None,
                        help="(Required for overlength) Maximum allowed sequence length")
    parser.add_argument("--log", "-L", type=str, default="log.txt",
                        help="(Optional) Log file for excluded sequences (default: log.txt)")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    arguments = parser.parse_args()
    if (arguments.overlength) and (arguments.size is None):
        sys.exit("When overlength, size is required")
    return arguments

def get_arguments():
    parser = parse_arguments()
        
    return {
        "input" : Path(parser.input),
        "output" : Path(parser.output),
        "overlength" : parser.overlength,
        "size" : parser.size,
        "log" : Path(parser.log)
    }

#Function which return unique sequences
def get_unique_seq_treated(input):
    """
    Generator that yields unique sequences from a FASTA file.
    Duplicate headers are skipped.
    Non-standard amino acids are replaced with 'X'.
    """
    #variables to control statements and pass to next function
    header = None
    names = set()
    unique = False
    seq_lines = []
    aa_set = set('ARNDCEQGHILKMFPSTWYVX') #create set with all valid amino acids 
    #open input file
    with open(input, 'r') as fin:
        
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
                    sequence_aa = '' #create empty variable to store new sequence
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

#Principal function of sequence
def main():
    #Getting arguments
    arguments = get_arguments()
    
    msg1 = '''#STEP1
                Eliminating  repeated sequences and not accepted amino acids '''
    print(msg1)

    log_file = None
    if arguments["overlength"]:
        log_path = Path(arguments["log"])
        if not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, 'a')
    with open(arguments["output"],'w') as out:
        msg2 = '''#STEP2
                Creating output file'''
        print(msg2)
        for header, sequences in get_unique_seq_treated(arguments["input"]):
            if arguments["overlength"]:
                seq_length = len(sequences.replace("\n",""))
                if seq_length < arguments["size"]:
                    out.write(f'>{header}\n{sequences}\n')
                else:
                    if log_file:
                        log_file.write(f'>{header} length={seq_length}\n{sequences}\n')
            else:
                out.write(f'>{header}\n{sequences}\n')
    msg3 = '''-------------------------------DONE--------------------------------'''
    print(msg3)
        

if __name__ == "__main__":
    main()


