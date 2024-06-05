import * as constants from './constants';
import { Container, Graphics } from 'pixi.js'

class Pellets {
    constructor(scaleX = 1, scaleY = 1) {
        this.pelletsList = [];
        this.pellets = new Graphics();
        this.scaleX = scaleX;
        this.scaleY = scaleY;
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
            let x = (pellet[0] + 1/2) * constants.TILEWIDTH * this.scaleX;
            let y = (pellet[1] + 1/2) * constants.TILEHEIGHT * this.scaleY;
            this.pellets.circle(x, y, 2*this.scaleX)
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

    updateScale(scaleX, scaleY) {
        this.scaleX = scaleX;
        this.scaleY = scaleY;
    }

}

export default Pellets;