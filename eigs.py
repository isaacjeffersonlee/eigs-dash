from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import numpy as np
import sympy as sp
import minpoly
import gunicorn


my_font = 'normal 20px monospace, Arial, sans-serif'


app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1('Eigenvalues and Eigenvectors', id='title'),
    html.H4('Author: Isaac Lee | Source: https://github.com/isaacjeffersonlee/eigs-dash',
        id='author-and-source-subtitle'),
    html.H2('Matrix Input', id='matrix-input-subtitle'),
    dcc.Textarea(
        id='matrix-input',
        value='[1 2 3; 0 1 5; 0 0 -2]',
        style={'width': '100%', 'height': 200, 'font': my_font}
    ),
    html.Button('Run', 
        id='run-button',
        n_clicks=0,
        className="my-button"
        # style={
        #     "background-color": "#000000",
        #     "border": "none",
        #     "color": "white",
        #     "padding": "10px 50px",
        #     "text-align": "center",
        #     "text-decoration": "none",
        #     "display": "inline-block",
        #     "font-size": "20px",
        #     "margin-top": "20px",
        #     "margin-bottom": "20px",
        #     'font': my_font
        # }
    ),
    dcc.Loading(
        id="loading-spinner",
        type="cube",
        color="black",
        children=[
            html.Div(id='textarea-output',
            style={'width': '100%',
                   'whiteSpace': 'pre',
                   'font': my_font}
            )
        ]
    )
])

@app.callback(
    Output(component_id='textarea-output', component_property='children'),
    Input(component_id='run-button', component_property='n_clicks'),
    State(component_id='matrix-input', component_property='value')
)
def update_output(n_clicks, value):
    # Prevent callback errors when we aren't finished inputing our matrix
    try:
        A = np.matrix(value)
        n, m = A.shape  # Get dimensions of the matrix
        if n != m:
            return "Matrix must be square!"
    except (ValueError, SyntaxError):
        return "Not a valid input! Only use Ints or Floats!"
    else:
        A = sp.Matrix(A)  # Conver to sympy matrix
        return_str = 'Input Matrix:\n'
        return_str += sp.pretty(A) + '\n' * 2

        x = sp.symbols('x')
        char_poly = A.charpoly(x).as_expr()
        return_str += "Characteristic Polynomial Expanded: "\
            f"{str(char_poly).replace('**', '^').replace('*', '')}" + '\n'
        return_str += "Characteristic Polynomial Factorized:"\
            f"{str(sp.factor(char_poly)).replace('**', '^').replace('*', '')}\n"

        try:
            min_poly = minpoly.get_min_poly(A)
            return_str += "Minimal Polynomial: "\
                f"{str(min_poly).replace('**', '^').replace('*', '')}" + '\n\n'
        except ValueError:
            return_str += "Minimal Polynomial could not be found!\n\n"

        eig_triplets = A.eigenvects()
        return_str += "(Eigen Value, Algebraic Multiplicity, Eigenspace Basis) triplets:" + '\n' * 2
        return_str += sp.pretty(eig_triplets, wrap_line=False) + '\n' * 2

        P, J = A.jordan_form()
        return_str += "P^-1 A P = J" + '\n' * 2
        return_str += sp.pretty([P**-1, A, P, J], wrap_line=False) + '\n' * 2

        return return_str


if __name__ == '__main__':
    app.run_server(debug=False)


