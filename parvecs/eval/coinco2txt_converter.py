'''
Used to convert the coinco (Kremer 2014) xml dataset format to a flat format.

Example of coinco format:
<document>
  <sent MASCfile="NYTnewswire9.txt" MASCsentID="s-r0" >
    <precontext>
    
    </precontext>
    <targetsentence>
    A mission to end a war
    </targetsentence>
    <postcontext>
    AUSTIN, Texas -- Tom Karnes was dialing for destiny, but not everyone wanted to cooperate.
    </postcontext>
    <tokens>
      <token id="XXX" wordform="A" lemma="a" posMASC="XXX" posTT="DT" />
      <token id="4" wordform="mission" lemma="mission" posMASC="NN" posTT="NN" problematic="no" >
        <substitutions>
          <subst lemma="calling" pos="NN" freq="1" />
          <subst lemma="campaign" pos="NN" freq="1" />
          <subst lemma="dedication" pos="NN" freq="1" />
          <subst lemma="devotion" pos="NN" freq="1" />
          <subst lemma="duty" pos="NN" freq="1" />
          <subst lemma="effort" pos="NN" freq="1" />
          <subst lemma="goal" pos="NN" freq="2" />
          <subst lemma="initiative" pos="NN" freq="1" />
          <subst lemma="intention" pos="NN" freq="1" />
          <subst lemma="movement" pos="NN" freq="1" />
          <subst lemma="plan" pos="NN" freq="2" />
          <subst lemma="pursuit" pos="NN" freq="1" />
          <subst lemma="quest" pos="NN" freq="1" />
          <subst lemma="step" pos="NN" freq="1" />
          <subst lemma="task" pos="NN" freq="2" />
        </substitutions>
      </token>

'''

import sys
import string
from xml.etree import ElementTree

to_wordnet_pos = {'N':'n','J':'a','V':'v','R':'r'}

def is_printable(s):
    return all(c in string.printable for c in s)


def clean_token(token):
    
    token = token.replace('&quot;', '"')
    token = token.replace('&apos;', "'")
    token = token.replace(chr(int("85",16)), "...")
    token = token.replace(chr(int("91",16)), "'")
    token = token.replace(chr(int("92",16)), "'")
    token = token.replace(chr(int("93",16)), '"')
    token = token.replace(chr(int("94",16)), '"')
    token = token.replace(chr(int("96",16)), '-')         
    if not is_printable(token):
        sys.stderr.write('TOKEN NOT PRINTABLE: '+''.join([str(c) for c in token if c in string.printable ]) + '\n')
        return "<UNK>"
    else:
        return token    

def subs2text(subs_element):
    subs = [(int(sub.attrib.get('freq')), clean_token(sub.attrib.get('lemma')).replace(';', ',')) for sub in subs_element.iter('subst')]  # sub.attrib.get('lemma').replace(';', ',') is used to fix a three cases in coinco where the lemma includes erroneously the char ';'. Since this char is used as a delimiter, we replace it with ','. 
    sorted_subs = sorted(subs, reverse=True)
    return ';'.join([sub + " " + str(freq) for freq, sub in sorted_subs])+';'

if __name__ == '__main__':
    
    if len(sys.argv) < 4:
        print "Usage: %s <input-coinco-filename> <output-test-filename> <output-gold-filename>" % sys.argv[0]
        sys.exit(1)
        
    with open(sys.argv[1], 'r') as f:
        coinco = ElementTree.parse(f)
    
    test_file = open(sys.argv[2], 'w') 
    gold_file = open(sys.argv[3], 'w') 
     
    sent_num = 0
    tokens_num = 0
        
    for sent in coinco.iter('sent'):
        sent_num += 1
        tokens = sent.find('tokens')
        sent_text = ""
        for token in tokens.iter('token'):
            sent_text = sent_text + clean_token(token.attrib.get('wordform')) + " "                
        sent_text = sent_text.strip()
        tok_position = -1
        for token in tokens.iter('token'):
            tok_position += 1
            if token.attrib.get('id') != 'XXX' and token.attrib.get('problematic') == 'no':
                tokens_num += 1
                try:
                    target_key = clean_token(token.attrib.get('lemma')) + '.' + to_wordnet_pos[token.attrib.get('posMASC')[0]]
                    test_file.write("%s\t%s\t%d\t%s\n" % (target_key, token.attrib.get('id'), tok_position, sent_text))
                    gold_file.write("%s %s :: %s\n" % (target_key, token.attrib.get('id'), subs2text(token.find('substitutions'))))
                except UnicodeEncodeError as e:
                    sys.stderr.write("ENCODING TARGET ERROR at token_id %s. %s\n" % (token.attrib.get('id'),e))
                    sys.exit(1)
                
    test_file.close()
    gold_file.close()
                           
    print 'Read %d sentences %d target tokens' % (sent_num, tokens_num)   
    
    
        
    