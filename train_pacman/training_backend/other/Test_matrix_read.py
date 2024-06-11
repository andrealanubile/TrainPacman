class TextTransformer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.grid = []
        self.transformed_text = ""

    def transform_text_with_spaces(self):
        with open(self.file_path, 'r') as file:
            content = file.read()
        
        translation_table = str.maketrans({
            'X': 'x', '0': 'x', '1': 'x', '2': 'x', '3': 'x',
            '4': 'x', '5': 'x', '6': 'x', '7': 'x', '8': 'x',
            '9': 'x', '=': 'x', 'n': ' ', '-': ' ', 'l': ' ',
            '.': '.', '+': '.', 'p': 'o', 'P': 'o'
        })
        
        self.transformed_text = content.translate(translation_table)
        
        # Create the grid, keeping the spaces intact
        self.grid = [list(line) for line in self.transformed_text.split('\n')]
        
        return self.transformed_text

    def save_transformed_text(self, output_file_path):
        with open(output_file_path, 'w') as output_file:
            output_file.write(self.transformed_text)

# Usage
file_path = 'maze1.txt'
output_file_path_with_spaces = 'transformed_maze1_with_spaces.txt'

transformer = TextTransformer(file_path)
transformed_text_with_spaces = transformer.transform_text_with_spaces()
transformer.save_transformed_text(output_file_path_with_spaces)

transformed_text_with_spaces, transformer.grid, output_file_path_with_spaces

print(transformed_text_with_spaces)