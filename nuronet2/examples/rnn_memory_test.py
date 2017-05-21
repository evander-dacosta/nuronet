import numpy
from collections import OrderedDict


class Instruction:
    """
    Takes a string and splits it into instructions and tokens
    """
    valid_isa = ['all', 'some', 'is', ';', '/', 'thus']

    def __init__(self, instruction):
        self.string = instruction
        self.instruction = self.split_instruction(instruction)
        self.tokens = self.get_tokens(self.instruction)
        self.isa = self.get_isa(self.instruction)
        self.template = self.get_template(self.instruction)
        
        self.check_validity(self.instruction)
        
    def get_tokens(self, instruction):
        tokens = []
        for word in instruction:
            if(word not in self.valid_isa + ['are']):
                if(word == 'are'):
                    word = 'is'
                tokens.append(word)
        return tokens
    
    def get_isa(self, instruction):
        if(len(self.tokens) == 0):
            self.tokens = self.get_tokens(instruction)
            return self.get_isa(instruction)
        return list(set(instruction) - set(self.tokens))
        
    def get_template(self, instruction):
        template = []
        for word in instruction:
            if(word in self.tokens):
                word = '{}'
            template.append(word)
        return template

    def check_validity(self, instruction):
        #check number of tokens is correct 
        if(not self.check_num_tokens()):
            print self.tokens, self.instruction
            raise Exception("Number of tokens in this instruction exceeds "
                            "the requirement of 2 per sentence. "
                            "Given {}".format(len(self.tokens)))
        #check order is correct
        if(not self.check_order()):
            raise Exception("The instruction is not semantically correct.")
        
    
    def check_num_tokens(self):
        if(len(self.tokens) == 2):
            return True
        return False
        
    def check_order(self):
        templates = ['all {} is {} ;', 'all {} are {} ;',
                     'some {} is {} ;', 'some {} are {} ;',
                     '{} is {} ;' , '{} are {} ;', 'thus all {} is {} /',
                     'thus some {} is {} /',
                     'thus all {} are {} /', 'thus some {} are {} /']
        templates = [t.split(' ') for t in templates]
        if(self.template in templates):
            return True
        return False

    @staticmethod
    def split_instruction(string):
        string = string.lower()
        final_symbol = None
        if(string[-1] == ';' or string[-1] == '/'):
            final_symbol = string[-1]
            string = string[:-1]

        string = string.strip()            
        words = string.split(' ')
        if(words[-1] == ' '):
            words = words[:-1]
        if(final_symbol is None):
            raise Exception("Instruction must have a terminating symbol. "
                            "Use either ';' or '/'")
        return words + [final_symbol]
        
    def to_binary(self, environment):
        binary_words = map(environment.word2binary, self.instruction)
        return binary_words
        
    @staticmethod
    def to_words(binary, environment):
        return map(environment.binary2word, binary)
        
    def __str__(self):
        return self.string
        
        
class Syllogism:
    def __init__(self, tokens, scramble=False):
        assert(isinstance(tokens, list) and len(tokens) == 3)
        self.scramble = scramble
        self.tokens = tokens
        self.instructions, all_some_order = self.make_instructions(tokens)
        self.truth = self.get_truth(all_some_order)
        
    def make_instructions(self, tokens):
        token_order = [(0, 1), (2, 0), (2, 1)]
        all_some_order = numpy.random.randint(0, 2, 3)
        
        instructions = self.all_some_templates(all_some_order)
        for i, j in enumerate(token_order):
            instructions[i] = instructions[i].format(tokens[j[0]],
                                                     tokens[j[1]])
        
        def make_instruction_object(instruction):
            return Instruction(instruction)
        
        instructions = map(make_instruction_object, instructions)
        if(self.scramble):
            numpy.random.shuffle(instructions)
        return instructions, all_some_order
        
    def all_some_templates(self, all_some_order):
        templates = [['all {} are {} ;', 'some {} are {} ;'],
                      ['all {} are {} ;', 'some {} are {} ;'],
                      ['thus all {} are {} /', 'thus some {} are {} /']]
        instructions = []
        for i, j in enumerate(all_some_order):
            instructions.append(templates[i][j])
        return instructions
    
    def get_truth(self, all_some_order):
        truths = {str([0, 0, 0]): True, str([0, 0, 1]): True, str([0, 1, 0]): False,
                  str([0, 1, 1]): False, str([1, 1, 1]): False, str([1, 1, 0]): False,
                   str([1, 0, 0]): False, str([1, 0, 1]): False}
        return truths[str(list(all_some_order))]

    def to_binary(self, environment):
        binary_instructions = [i.to_binary(environment) for i in self.instructions]
        binary_syl = []
        for instruction in binary_instructions:
            binary_syl += instruction
        return binary_syl        
        
    @staticmethod
    def to_words(binary, environment):
        return [Instruction.to_words(b, environment) for b in binary]
        
    def __str__(self):
        return "\n".join([x.string for x in self.instructions])
            
        

class Environment:
    def __init__(self, bit_length=20):
        self.bit_length = bit_length
        self.isa = ['all', 'some', 'is', ';', '/', 'thus']
        self.tokens = self.generate_tokens()
        self.binary_map = self.create_binary_map()
        
    def generate_tokens(self):
        n_tokens = self.bit_length - len(self.isa)
        token_lengths = numpy.random.randint(3, 7, n_tokens)
        tokens = map(self.generate_one_token, token_lengths)
        
        #remove duplicates
        tokens = list(OrderedDict.fromkeys(tokens))
        return tokens

    def generate_one_token(self, token_length):
        letters = numpy.array(list('abcdefghijklmnopqrstuvwxyz'))
        indices = numpy.random.randint(0, len(letters), token_length)
        return ''.join(list(letters[indices]))
        
    def create_binary_map(self):
        all_words = self.isa + self.tokens
        bin_map = []
        for i, word in enumerate(all_words):
            binary = numpy.zeros((self.bit_length,))
            binary[i] = 1
            bin_map.append((word, list(binary)))
        return OrderedDict(bin_map)
        
    def word2binary(self, word):
        if(word == 'are'):
            word = 'is'
        return self.binary_map[word]
        
    def binary2word(self, binary):
        index = numpy.array(binary).nonzero()[0][0]
        return self.binary_map.keys()[index]
        
    def get_unique_tokens(self, n=3):
        numbers = numpy.random.randint(0, len(self.tokens), 50)
        numbers = numpy.unique(numbers)
        numpy.random.shuffle(numbers)
        return list(numpy.array(self.tokens)[numbers[:n]])
        

        
        
class SyllogismGenerator:
    def __init__(self, n_datapoints, n_tokens=20, scramble=False):
        """
        scramble shuffles the order of the instructions in a syllogism
        """
        self.scramble=scramble
        self.n = n_datapoints
        self.environment = Environment(n_tokens)
        self.syllogisms = self.generate_syllogisms(n_datapoints)
        
    def generate_syllogisms(self, n):
        return [Syllogism(self.environment.get_unique_tokens(3), self.scramble) for _ in range(n)]

    def __getitem__(self, n):
        return self.syllogisms[n]
        
    def to_binary(self, syllogisms):
        dataset = []
        truths = []
        if(not isinstance(syllogisms, list)):
            syllogisms = [syllogisms]
        for syllogism in syllogisms:
            dataset += [syllogism.to_binary(self.environment)]
            truths += [[syllogism.truth]]
        dataset = numpy.array(dataset).astype(numpy.float32)
        truths = numpy.array(truths).astype(numpy.float32)
        return dataset, truths
        
        
    def get_data(self, test_set=0.1, scramble=False):
        """
        Scramble randomises the order of the instructions
        in the syllogism
        """
        dataset, truths = self.to_binary(self.syllogisms)

        
        n_test = int(self.n * test_set)
        x_test = dataset[-n_test:]
        y_test = truths[-n_test:]
        x = dataset[:-n_test]
        y = truths[:-n_test]
        return x, y, x_test, y_test
        
    def test_model(self, model, scramble=False):
        syllogism = Syllogism(self.environment.get_unique_tokens(3), scramble)
        print syllogism, syllogism.truth
        binary, _ = self.to_binary(syllogism)
        print "Model prediction is: ", model.predict(binary) > 0.5

        

if __name__ == "__main__":
    from nuronet2.base import NeuralNetwork
    from nuronet2.layers import RNNLayer, DenseLayer
    datagen = SyllogismGenerator(5000, scramble=True)
    x, y, x_test, y_test = datagen.get_data()
    
    n_epochs = 30
    model = NeuralNetwork()
    model.add(RNNLayer(64, return_sequences=False, input_shape=(16, 20),
                       go_backwards=True, w_dropout=0.2))
    model.add(DenseLayer(1, activation="sigmoid"))
    model.compile("adam", "binary_crossentropy")
    model.fit(x, y, n_epochs=n_epochs)
    
    predictions = model.predict(x_test) > 0.5
    matches = (predictions == y_test)
    score = sum(matches)[0] / float(len(matches))
    print score * 100.
    