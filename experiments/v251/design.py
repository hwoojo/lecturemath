"""Conditions fixed before reading V2.51 predictions."""
CONTROLS=[
 ('eq','factorial',0,r'3=3'),
 ('factorial','factorial',1,r'3!=6'),
 ('indexed_factorial','factorial',2,r'|S_3|=3!=6'),
 ('general_factorial','factorial',3,r'|S_n|=n!'),
 ('frac','fraction',0,r'\frac{a}{b}'),
 ('frac_sum','fraction',1,r'\frac{a+b}{c-d}'),
 ('nested_frac','fraction',2,r'\frac{1}{1+\frac{1}{x}}'),
 ('long_fraction','fraction',3,r'y=\frac{3-x}{2}=\frac{3-\frac{1}{3}}{2}'),
 ('vec2','vector',0,r'\begin{bmatrix}1\\2\end{bmatrix}'),
 ('vec3','vector',1,r'\begin{bmatrix}1\\2\\3\end{bmatrix}'),
 ('vec_symbolic','vector',2,r'\begin{bmatrix}p\\q\\r\end{bmatrix}'),
 ('matrix22','matrix',0,r'\begin{bmatrix}1&2\\3&4\end{bmatrix}'),
 ('matrix23','matrix',1,r'\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}'),
 ('matrix33','matrix',2,r'\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix}'),
 ('augmented','matrix',3,r'\left[\begin{array}{cc|c}1&-3&0\\2&4&0\end{array}\right]'),
 ('line1','multiline',0,r'3x=1'),
 ('lines2','multiline',1,r'\begin{aligned}3x&=1\\x+2y&=3\end{aligned}'),
 ('lines3','multiline',2,r'\begin{aligned}x+y&=2\\x-y&=0\\z&=3\end{aligned}'),
 ('sup','scripts',0,r'x^2'),
 ('sub','scripts',1,r'x_i'),
 ('sub_sup','scripts',2,r'x_i^2+y_j^3'),
 ('complex_scripts','scripts',3,r'x_{i+1}^{n-1}+a_{ij}^{(k)}'),
 ('set','set_text',0,r'S=\{1,2,3\}'),
 ('set_text','set_text',1,r'S_3=\{A_{3\times3}\mid A\text{ is a permutation matrix}\}'),
]

# These five are diagnostic development cases, not independent evaluation.
SENSITIVITY=[
 ('lecture_01_10423','formula_system'),
 ('lecture_06_47799','augmented_matrix'),
 ('lecture_07_67415','symbolic_vector'),
 ('lecture_08_53016','permutation_matrix'),
 ('lecture_08_53016','cardinality_factorial'),
]
ALTERNATE_PRINTS=['vec3','matrix33','factorial','indexed_factorial','lines2','set_text']
SPLIT_ROWS={
 'lines2':[r'3x=1',r'x+2y=3'],
 'lines3':[r'x+y=2',r'x-y=0',r'z=3'],
 # Matrix splitting is deliberately destructive, diagnostic only.
 'matrix22':[r'1\quad2',r'3\quad4'],
}
