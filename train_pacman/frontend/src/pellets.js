import * as constants from './constants';
import { Container, Graphics } from 'pixi.js'

class Pellets {
    constructor() {
        this.pelletsList = [];
        this.pellets = new Graphics();
    }

    // readPelletsFile(file) {
    //     return file.trim().split("\n").map(line => line.trim().split(/\s+/));
    // }

    setPelletList(pelletList) {
        this.pelletList = pelletList;
    }


    drawPellets() {
        this.pellets.clear()
        let pellet;
        for (pellet of this.pelletList) {
            let x = (pellet[0] + 1/2) * constants.TILEWIDTH;
            let y = (pellet[1] + 1/2) * constants.TILEHEIGHT;
            this.pellets.circle(x, y, 2)
            this.pellets.fill(0xffffff)
        }


        // for (let row = 0; row < this.data.length; row++) {
        //     for (let col = 0; col < this.data[row].length; col++) {
        //         if (this.data[row][col] == 1) {
        //             let x = (col + 1/2) * constants.TILEWIDTH;
        //             let y = (row + 1/2) * constants.TILEHEIGHT;
        //             pellets.circle(x, y, 2)
        //             pellets.fill(0xffffff)
        //         }
        //     }
        // }
        // return pellets;
    }

}

export default Pellets;