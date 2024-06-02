import * as constants from './constants'
import Spritesheet from './spritesheet';
import { Sprite } from 'pixi.js'

class GhostSprites extends Spritesheet {
    constructor(name, mode, direction) {
        super();
        this.name = name;
        this.mode = mode;
        this.direction = direction;
        this.ghostMap = {BLINKY: 0, PINKY: 2, INKY: 4, CLYDE: 6};
        this.image = this.getStartImage();
    }

    drawGhost() {
        let gloc = this.ghostMap[this.name];
        if (this.mode in [constants.SCATTER, constants.CHASE]) {
            switch (this.direction) {
                case constants.LEFT:
                    this.image = this.getImage(gloc, 8);
                    break;
                case constants.RIGHT:
                    this.image = this.getImage(gloc, 10);
                    break;
                case constants.DOWN:
                    this.image = this.getImage(gloc, 6);
                    break;
                case constants.UP:
                    this.image = this.getImage(gloc, 4);
                    break;
            }
        } else if (this.mode == constants.FRIGHT) {
            this.image = this.getImage(10, 4);
        } else if (this.mode == constants.SPAWN) {
            switch (this.direction) {
                case constants.LEFT:
                    this.image = this.getImage(8, 8)
                case constants.RIGHT:
                    this.image = this.getImage(8, 10);
                    break;
                case constants.DOWN:
                    this.image = this.getImage(8, 6);
                    break;
                case constants.UP:
                    this.image = this.getImage(8, 4);
                    break;
            }
        }
        let col = 6;
        let row = 5;
        let ghostSprite = new Sprite(this.image);
        ghostSprite.anchor.set(0.5);
        ghostSprite.x = (col + 1/2) * constants.TILEWIDTH;
        ghostSprite.y = (row + 1/2) * constants.TILEHEIGHT;;
        return ghostSprite;
    }

    getStartImage() {
        return this.getImage(this.ghostMap[this.name], 4)
    }


    getImage(x, y) {
        return super.getImage(Math.floor(x/2), Math.floor(y/2), 2*constants.TILEWIDTH, 2*constants.TILEHEIGHT);
    }
}

export default GhostSprites;