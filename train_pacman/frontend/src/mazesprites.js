import Spritesheet from './spritesheet';
import { Assets, Container, Sprite } from 'pixi.js'
import * as constants from './constants';

class MazeSprites extends Spritesheet {
    constructor(mazefile, rotfile, scaleX = 1, scaleY = 1) {
        super();
        this.data = this.readMazeFile(mazefile);
        this.rotdata = this.readMazeFile(rotfile);
        this.scaleX = scaleX;
        this.scaleY = scaleY;
        this.background = new Container();
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

    getBackground() {
        return this.background;
    }

    constructBackground(y) {
        for (let row = 0; row < this.data.length; row++) {
            for (let col = 0; col < this.data[row].length; col++) {
                if (this.isNumeric(this.data[row][col])) {
                    let x = parseInt(this.data[row][col]) + 12;
                    let texture = this.getImage(x, y);
                    let sprite = new Sprite(texture);
                    let rotval = parseInt(this.rotdata[row][col]);
                    sprite.anchor.set(0.5);
                    sprite.rotation = -rotval * Math.PI / 2;
                    sprite.x = (col + 1/2) * constants.TILEWIDTH * this.scaleX;
                    sprite.y = (row + 1/2) * constants.TILEHEIGHT * this.scaleY;
                    sprite.scale.set(this.scaleX, this.scaleY);
                    this.background.addChild(sprite);
                } else if (this.data[row][col] === '=') {
                    let sprite = new Sprite(this.getImage(10, 8));
                    sprite.x = col * constants.TILEWIDTH * this.scaleX;
                    sprite.y = row * constants.TILEHEIGHT * this.scaleY;
                    sprite.scale.set(this.scaleX, this.scaleY);
                    this.background.addChild(sprite);
                }
            }
        }
    }

    updateScale(scaleX, scaleY) {
        this.scaleX = scaleX;
        this.scaleY = scaleY;
    }

    updateBackground() {
        if (this.background) {
            this.background.removeChildren();
            this.constructBackground(0);
        }
    }
}

export default MazeSprites;