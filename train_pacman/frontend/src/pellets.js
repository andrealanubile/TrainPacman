import * as constants from './constants';
import { Container, Graphics } from 'pixi.js'

class Pellets {
    constructor(pelletsFile) {
        this.data = this.readPelletsFile(pelletsFile);
    }

    readPelletsFile(file) {
        return file.trim().split("\n").map(line => line.trim().split(/\s+/));
    }


    drawPellets() {
        let pellets = new Graphics()

        for (let row = 0; row < this.data.length; row++) {
            for (let col = 0; col < this.data[row].length; col++) {
                if (this.data[row][col] == 1) {
                    let x = (col + 1/2) * constants.TILEWIDTH;
                    let y = (row + 1/2) * constants.TILEHEIGHT;
                    pellets.circle(x, y, 2)
                    pellets.fill(0xffffff)
                }
            }
        }
        return pellets;
    }

}

export default Pellets;