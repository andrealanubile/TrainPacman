import Spritesheet from './spritesheet';
import { Assets, Container, Sprite } from 'pixi.js'
import * as constants from './constants';

class MazeSprites extends Spritesheet {
    constructor(mazefile, rotfile) {
        super();
        this.data = this.readMazeFile(mazefile);
        this.rotdata = this.readMazeFile(rotfile);
    }

    readMazeFile(file) {
        return file.trim().split("\n").map(line => line.trim().split(/\s+/));
    }

    getImage(x, y) {
        return super.getImage(x, y, constants.TILEWIDTH, constants.TILEHEIGHT);
    }

    isNumeric(str) {
        if (typeof str != "string") return false
        return !isNaN(str) && !isNaN(parseFloat(str)) 
    }

    constructBackground(y) {
        let background = new Container();

        for (let row = 0; row < this.data.length; row++) {
            for (let col = 0; col < this.data[row].length; col++) {
                if (this.isNumeric(this.data[row][col])) {
                    let x = parseInt(this.data[row][col]) + 12;
                    let texture = this.getImage(x, y);
                    let sprite = new Sprite(texture);
                    let rotval = parseInt(this.rotdata[row][col]);
                    sprite.anchor.set(0.5);
                    sprite.rotation = -rotval * Math.PI / 2;
                    sprite.x = (col + 1/2) * constants.TILEWIDTH;
                    sprite.y = (row + 1/2) * constants.TILEHEIGHT;
                    background.addChild(sprite);
                } else if (this.data[row][col] === '=') {
                    let sprite = new Sprite(this.getImage(10, 8));
                    sprite.x = col * constants.TILEWIDTH;
                    sprite.y = row * constants.TILEHEIGHT;
                    background.addChild(sprite);
                }
            }
        }

        return background;
    }
}

export default MazeSprites;