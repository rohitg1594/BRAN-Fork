import urllib.request as urllib
import requests
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pid', type=int, default=1329249,
                    help='an integer for the accumulator')
parser.add_argument('--out_file', type=str, help='output file for pubtator')
args = parser.parse_args()
pid = args.pid
ncbi_url = "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/BioConcept/"

with open(args.out_file, "w") as f:
    url_submit =  ncbi_url + str(pid) + "/PubTator/"
    print('created url')
    try: 
        urllib_result = requests.get(url_submit, timeout=1)
    except requests.exceptions.Timeout:
        sys.exit(1)
    print('got result')
    f.write(urllib_result.text[:-1])
    #for path, (ent1_id, ent2_id) in values:
    #    f.write('{}\t{}\t{}\t{}\n'.format(pid, theme, ent1_id, ent2_id))
    f.write('{}\t{}\t{}\t{}\n'.format(pid, "TEST_REL", "TEST_ID1", "TEST_ID2"))
    f.write('\n')
