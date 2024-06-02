import * as constants from './constants'
import Spritesheet from './spritesheet';
import { Container, Sprite, spritesheetAsset } from 'pixi.js'

class LifeSprites extends Spritesheet {
    constructor() {
        super();
        this.images = [];
    }

    removeImage() {
        if (this.images.length > 0) {
            this.images.shift();
        }
    }

    resetLives(numLives) {
        this.images = [];
        for (let i = 0; i < numLives; i++) {
            this.images.push(this.getImage(0, 0))
        }
    }

    getImage(x, y) {
        return super.getImage(Math.floor(x/2), Math.floor(y/2), 2*constants.TILEWIDTH, 2*constants.TILEHEIGHT);
    }

    drawLives() {
        let livesContainer = new Container();
        for (let i = 0; i < this.images.length; i++) {
            let lifeSprite = new Sprite(this.images[i]);
            lifeSprite.x = i * constants.TILEWIDTH * 2;
            lifeSprite.y = constants.SCREENHEIGHT - constants.TILEWIDTH * 2;
            livesContainer.addChild(lifeSprite);
        }
        return livesContainer;
    }

}

export default LifeSprites;