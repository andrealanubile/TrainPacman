const path = require('path');
const CopyWebpackPlugin = require('copy-webpack-plugin')
const HTMLWebpackPlugin = require('html-webpack-plugin')
const TerserPlugin = require('terser-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const Dotenv = require('dotenv-webpack')

module.exports = {
    mode: 'production',
    entry: './src/index.js',
    output: {
        filename: 'bundle.[contenthash].js',
        path: path.resolve(__dirname, 'dist')
    },
    module: {
        rules: [{
            test: /\.(js)$/,
            exclude: /node_modules/,
            use: {
                loader: 'babel-loader'
            }
        },
        {
            test: /\.css$/,
            use: ['style-loader', 'css-loader'],
        },
    ]
    },
    optimization: {
        minimizer: [new TerserPlugin({
            terserOptions: {
                format: {
                    comments: false //use it for removing comments like "/*! ... */"
                }
            },
            extractComments: false
        })]
    },
    plugins: [
        new CleanWebpackPlugin(),
        new CopyWebpackPlugin({
            patterns: [
                { from: 'build/assets', to: 'assets' }
            ]
        }),
        new HTMLWebpackPlugin({
            template: 'src/index.html',
            filename: 'index.html',
            hash: true,
            minify: {
                collapseWhitespace: true, // Ensures HTML is minified
                removeComments: true      // Removes HTML comments
            }
        }),
        new Dotenv(),
    ]
}